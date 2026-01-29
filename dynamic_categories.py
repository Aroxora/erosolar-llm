#!/usr/bin/env python3
"""
DYNAMIC CATEGORY DISCOVERY & EXPANSION
=======================================
Automatically discovers, expands, and maintains comprehensive training categories.

Features:
1. Extract topics from existing training data
2. LLM-powered category/seed expansion
3. Hierarchical category discovery (domain -> subdomain -> seeds)
4. Gap detection and filling
5. Wikipedia/knowledge base integration

Usage:
    python dynamic_categories.py --discover          # Discover from training data
    python dynamic_categories.py --expand            # LLM-expand all categories
    python dynamic_categories.py --expand-domain space  # Expand specific domain
    python dynamic_categories.py --auto              # Full auto-discovery pipeline
"""

import os
import json
import asyncio
import aiohttp
import re
import random
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple
import argparse

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Use gpt-5.1-codex-mini for optimal expansion (consistent with training data generation)
MODEL = os.environ.get("EXPANSION_MODEL", "gpt-5.1-codex-mini")
RESPONSE_API = "https://api.openai.com/v1/responses"  # Response API for codex models
CHAT_API = "https://api.openai.com/v1/chat/completions"  # Fallback for chat models
CHAT_MODELS = {"gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"}

DATA_STORE = Path("data_store")
DYNAMIC_CATEGORIES_FILE = DATA_STORE / "dynamic_categories.json"
DISCOVERED_SEEDS_FILE = DATA_STORE / "discovered_seeds.json"
SEED_USAGE_FILE = DATA_STORE / "seed_usage.json"

# High-level domains to auto-expand (comprehensive knowledge coverage)
ROOT_DOMAINS = [
    # STEM
    "mathematics", "physics", "chemistry", "biology", "astronomy", "geology",
    "computer_science", "engineering", "medicine", "environmental_science",

    # Social Sciences
    "psychology", "sociology", "economics", "political_science", "anthropology",
    "linguistics", "education", "law", "criminology",

    # Humanities
    "history", "philosophy", "literature", "art", "music", "religion",
    "archaeology", "cultural_studies",

    # Applied/Professional
    "business", "finance", "marketing", "management", "accounting",
    "healthcare", "nursing", "public_health", "nutrition",

    # Technology
    "programming", "web_development", "mobile_development", "devops",
    "cybersecurity", "data_science", "machine_learning", "cloud_computing",
    "blockchain", "iot", "robotics", "game_development",

    # Geography & Places
    "world_countries", "us_states", "world_cities", "landmarks", "geography",

    # Practical Skills
    "cooking", "gardening", "home_improvement", "automotive", "crafts",
    "photography", "fitness", "sports", "outdoor_activities",

    # Life & Society
    "parenting", "relationships", "career", "personal_finance", "travel",
    "pets", "fashion", "entertainment", "gaming",

    # Military & Defense
    "military_history", "military_branches", "military_technology",
    "defense_systems", "military_strategy",

    # Space & Exploration
    "space_exploration", "planets", "space_technology", "astronauts",
    "space_agencies", "satellites", "space_missions",

    # Industry Specific
    "agriculture", "manufacturing", "energy", "transportation", "logistics",
    "real_estate", "hospitality", "retail", "telecommunications",
]

# ════════════════════════════════════════════════════════════════════════════════
# SEED USAGE TRACKER - Persistent tracking of used seeds
# ════════════════════════════════════════════════════════════════════════════════

class SeedUsageTracker:
    """
    Tracks which seeds have been used in training data generation.
    Prioritizes unused seeds to maximize diversity.
    Persistently stores usage data with hashes for efficient lookup.
    """

    def __init__(self, storage_path: Path = SEED_USAGE_FILE):
        self.storage_path = storage_path
        self.usage_data: Dict[str, Dict] = {}  # category -> {seed_hash: usage_count}
        self.seed_to_hash: Dict[str, str] = {}  # seed_text -> hash
        self.hash_to_seed: Dict[str, str] = {}  # hash -> seed_text
        self._load()

    def _hash_seed(self, seed: str, category: str) -> str:
        """Generate unique hash for a seed in a category."""
        import hashlib
        content = f"{category}:{seed.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load(self):
        """Load usage data from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    self.usage_data = data.get("usage", {})
                    self.seed_to_hash = data.get("seed_to_hash", {})
                    self.hash_to_seed = data.get("hash_to_seed", {})
                    print(f"[SeedTracker] Loaded {sum(len(v) for v in self.usage_data.values())} seed usage records")
            except Exception as e:
                print(f"[SeedTracker] Error loading: {e}")

    def save(self):
        """Save usage data to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump({
                "usage": self.usage_data,
                "seed_to_hash": self.seed_to_hash,
                "hash_to_seed": self.hash_to_seed,
                "total_uses": sum(sum(v.values()) for v in self.usage_data.values()),
                "unique_seeds": sum(len(v) for v in self.usage_data.values()),
            }, f, indent=2)

    def mark_used(self, seed: str, category: str, count: int = 1):
        """Mark a seed as used."""
        seed_hash = self._hash_seed(seed, category)

        if category not in self.usage_data:
            self.usage_data[category] = {}

        if seed_hash not in self.usage_data[category]:
            self.usage_data[category][seed_hash] = 0

        self.usage_data[category][seed_hash] += count
        self.seed_to_hash[f"{category}:{seed}"] = seed_hash
        self.hash_to_seed[seed_hash] = seed

    def get_usage_count(self, seed: str, category: str) -> int:
        """Get how many times a seed has been used."""
        seed_hash = self._hash_seed(seed, category)
        return self.usage_data.get(category, {}).get(seed_hash, 0)

    def is_used(self, seed: str, category: str) -> bool:
        """Check if a seed has been used at all."""
        return self.get_usage_count(seed, category) > 0

    def prioritize_seeds(self, seeds: List[str], category: str) -> List[str]:
        """
        Sort seeds to prioritize unused ones first.
        Returns seeds sorted by usage count (lowest first).
        """
        seed_usage = [(seed, self.get_usage_count(seed, category)) for seed in seeds]
        # Sort by usage count, then randomize within same count
        random.shuffle(seed_usage)  # Randomize first
        seed_usage.sort(key=lambda x: x[1])  # Then stable sort by count
        return [seed for seed, _ in seed_usage]

    def get_unused_seeds(self, seeds: List[str], category: str) -> List[str]:
        """Get only seeds that haven't been used yet."""
        return [s for s in seeds if not self.is_used(s, category)]

    def get_least_used_seeds(self, seeds: List[str], category: str, n: int = None) -> List[str]:
        """Get the n least-used seeds."""
        prioritized = self.prioritize_seeds(seeds, category)
        if n:
            return prioritized[:n]
        return prioritized

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        stats = {
            "categories": len(self.usage_data),
            "total_unique_seeds": sum(len(v) for v in self.usage_data.values()),
            "total_uses": sum(sum(v.values()) for v in self.usage_data.values()),
            "by_category": {},
        }

        for cat, seeds in self.usage_data.items():
            stats["by_category"][cat] = {
                "unique_seeds": len(seeds),
                "total_uses": sum(seeds.values()),
                "max_uses": max(seeds.values()) if seeds else 0,
            }

        return stats

    def get_coverage(self, seeds: List[str], category: str) -> Tuple[int, int, float]:
        """
        Get coverage statistics for a seed list.
        Returns: (used_count, total_count, percentage)
        """
        used = sum(1 for s in seeds if self.is_used(s, category))
        total = len(seeds)
        pct = (used / total * 100) if total > 0 else 0
        return used, total, pct

    def reset_category(self, category: str):
        """Reset usage tracking for a category."""
        if category in self.usage_data:
            del self.usage_data[category]
            print(f"[SeedTracker] Reset usage for category: {category}")

    def reset_all(self):
        """Reset all usage tracking."""
        self.usage_data = {}
        self.seed_to_hash = {}
        self.hash_to_seed = {}
        print("[SeedTracker] Reset all usage tracking")


# Global tracker instance
_seed_tracker: Optional[SeedUsageTracker] = None

def get_seed_tracker() -> SeedUsageTracker:
    """Get the global seed tracker instance."""
    global _seed_tracker
    if _seed_tracker is None:
        _seed_tracker = SeedUsageTracker()
    return _seed_tracker


# ════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamicCategory:
    """A dynamically discovered/expanded category."""
    name: str
    parent: Optional[str] = None
    templates: List[str] = field(default_factory=list)
    seeds: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    subcategories: List[str] = field(default_factory=list)
    source: str = "auto"  # auto, llm, extracted, manual
    confidence: float = 1.0

    def to_domain_format(self) -> dict:
        """Convert to DOMAINS dict format."""
        return {
            "templates": self.templates or self._default_templates(),
            "seeds": self.seeds,
            "tasks": self.tasks,
        }

    def _default_templates(self) -> List[str]:
        """Generate default templates based on category type."""
        return [
            f"Explain {{seed}}",
            f"What is {{seed}}?",
            f"Tell me about {{seed}}",
        ]


@dataclass
class DiscoveryStats:
    """Statistics from discovery process."""
    categories_discovered: int = 0
    seeds_discovered: int = 0
    from_training_data: int = 0
    from_llm_expansion: int = 0
    from_hierarchical: int = 0


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING DATA EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

class TrainingDataExtractor:
    """Extract topics and entities from existing training data."""

    def __init__(self):
        self.entity_patterns = [
            # Capitalized phrases (proper nouns)
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            # Technical terms
            r'\b((?:API|HTTP|SQL|JSON|XML|REST|GraphQL|TCP|IP|DNS|SSH|SSL|TLS)\b)',
            # Programming languages
            r'\b(Python|JavaScript|TypeScript|Java|C\+\+|Rust|Go|Ruby|PHP|Swift|Kotlin)\b',
            # Frameworks
            r'\b(React|Vue|Angular|Django|Flask|FastAPI|Express|Spring|Rails)\b',
        ]

    def extract_from_jsonl(self, filepath: Path) -> Dict[str, Set[str]]:
        """Extract entities from JSONL training data."""
        entities = defaultdict(set)

        if not filepath.exists():
            return entities

        with open(filepath) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    messages = record.get("messages", [])
                    category = record.get("metadata", {}).get("category", "unknown")

                    for msg in messages:
                        content = msg.get("content", "")
                        extracted = self._extract_entities(content)
                        entities[category].update(extracted)
                except:
                    continue

        return entities

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract potential seed entities from text."""
        entities = set()

        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match) > 2 and len(match) < 50:
                    entities.add(match)

        return entities

    def extract_topics_from_prompts(self, filepath: Path) -> Counter:
        """Extract common topics from user prompts."""
        topics = Counter()

        if not filepath.exists():
            return topics

        # Common topic indicators
        topic_patterns = [
            r'(?:explain|describe|what is|how does|tell me about)\s+(.+?)(?:\?|$)',
            r'(?:write|create|implement)\s+(?:a|an)?\s*(.+?)(?:\s+(?:that|which|for)|$)',
        ]

        with open(filepath) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    messages = record.get("messages", [])
                    for msg in messages:
                        if msg.get("role") == "user":
                            content = msg.get("content", "").lower()
                            for pattern in topic_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                for match in matches:
                                    if len(match) > 3:
                                        topics[match.strip()] += 1
                except:
                    continue

        return topics


# ════════════════════════════════════════════════════════════════════════════════
# LLM-POWERED EXPANSION
# ════════════════════════════════════════════════════════════════════════════════

class LLMExpander:
    """Use LLM to expand categories and discover seeds."""

    EXPANSION_PROMPT = """You are a knowledge categorization expert. Given a domain/topic, generate comprehensive subcategories and seed terms.

Domain: {domain}
{context}

Generate a JSON response with:
1. "subcategories": List of 5-15 major subcategories within this domain
2. "seeds": List of 20-50 specific terms/concepts/entities in this domain
3. "templates": List of 3-5 question templates using {{seed}} placeholder
4. "related_domains": List of 3-5 related domains to explore

Requirements:
- Seeds should be specific, concrete terms (not generic)
- Include proper nouns, technical terms, key concepts
- Be comprehensive - cover breadth of the domain
- Subcategories should be distinct and non-overlapping

Respond ONLY with valid JSON, no other text."""

    HIERARCHICAL_PROMPT = """Expand the following category into detailed seeds.

Category: {category}
Parent Domain: {parent}

Generate 30-100 specific items/terms/examples in this category.
These should be:
- Concrete and specific (not generic descriptions)
- Factually accurate
- Diverse in coverage
- Properly spelled and formatted

Respond with a JSON object: {{"seeds": ["item1", "item2", ...]}}"""

    def __init__(self):
        self.session = None

    async def expand_domain(self, domain: str, context: str = "") -> Optional[DynamicCategory]:
        """Expand a domain using LLM."""
        if not API_KEY:
            print("Warning: No API key, skipping LLM expansion")
            return None

        prompt = self.EXPANSION_PROMPT.format(
            domain=domain,
            context=f"Context: {context}" if context else ""
        )

        result = await self._call_llm(prompt)
        if not result:
            return None

        try:
            data = json.loads(result)
            return DynamicCategory(
                name=domain,
                templates=data.get("templates", []),
                seeds=data.get("seeds", []),
                subcategories=data.get("subcategories", []),
                source="llm",
            )
        except json.JSONDecodeError:
            return None

    async def expand_hierarchical(self, category: str, parent: str) -> List[str]:
        """Expand a specific category with detailed seeds."""
        if not API_KEY:
            return []

        prompt = self.HIERARCHICAL_PROMPT.format(
            category=category,
            parent=parent
        )

        result = await self._call_llm(prompt)
        if not result:
            return []

        try:
            data = json.loads(result)
            return data.get("seeds", [])
        except:
            return []

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API. Auto-selects Response API or Chat API based on model."""
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        # Use Chat API for chat models, Response API for codex models
        use_chat_api = MODEL in CHAT_MODELS

        if use_chat_api:
            api_url = CHAT_API
            payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.7,
            }
        else:
            # Response API for codex models (gpt-5.1-codex-mini)
            api_url = RESPONSE_API
            payload = {
                "model": MODEL,
                "input": prompt,
                "max_output_tokens": 4096,
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        print(f"API Error ({resp.status}): {error[:200]}")
                        return None
                    data = await resp.json()

                    # Parse response based on API type
                    if use_chat_api:
                        return data["choices"][0]["message"]["content"]
                    else:
                        # Response API format
                        output = data.get("output", [])
                        content = ""
                        for item in output:
                            if item.get("type") == "message":
                                for c in item.get("content", []):
                                    if c.get("type") == "output_text":
                                        content += c.get("text", "")
                        return content or data.get("output_text", "")

        except Exception as e:
            print(f"LLM call failed: {e}")
            return None


# ════════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE SEEDS (Pre-computed comprehensive lists)
# ════════════════════════════════════════════════════════════════════════════════

# These are exhaustive pre-computed seed lists for key domains
# No LLM call needed - just use directly

KNOWLEDGE_BASE = {
    # ─────────────────────────────────────────────────────────────────────────────
    # MILITARY & DEFENSE
    # ─────────────────────────────────────────────────────────────────────────────
    "military_branches": {
        "templates": ["Tell me about {seed}", "What is the role of {seed}?", "History of {seed}"],
        "seeds": [
            # US Military
            "US Army", "US Navy", "US Air Force", "US Marine Corps", "US Coast Guard",
            "US Space Force", "National Guard", "Army Reserve", "Navy SEALs", "Delta Force",
            "Green Berets", "Army Rangers", "Marine Raiders", "Air Force Pararescue",
            # Other Countries
            "British Army", "Royal Navy", "Royal Air Force", "SAS", "SBS",
            "French Foreign Legion", "Bundeswehr", "IDF", "Russian Spetsnaz",
            "Chinese PLA", "Indian Army", "Japanese Self-Defense Forces",
            "Australian Defence Force", "Canadian Armed Forces", "NATO forces",
        ],
        "tasks": []
    },
    "military_equipment": {
        "templates": ["Explain {seed}", "How does {seed} work?", "Specifications of {seed}"],
        "seeds": [
            # Aircraft
            "F-22 Raptor", "F-35 Lightning II", "F-16 Fighting Falcon", "F-15 Eagle",
            "B-2 Spirit", "B-52 Stratofortress", "C-130 Hercules", "V-22 Osprey",
            "AH-64 Apache", "UH-60 Black Hawk", "CH-47 Chinook", "MQ-9 Reaper",
            "A-10 Thunderbolt II", "E-3 Sentry AWACS", "KC-135 Stratotanker",
            # Naval
            "Aircraft carrier", "Destroyer", "Cruiser", "Submarine", "Frigate",
            "USS Nimitz", "USS Gerald R. Ford", "Virginia-class submarine",
            "Ohio-class submarine", "Arleigh Burke-class destroyer",
            # Ground
            "M1 Abrams", "Bradley Fighting Vehicle", "Stryker", "MRAP", "Humvee",
            "M777 Howitzer", "HIMARS", "Patriot missile", "THAAD", "Javelin missile",
            "TOW missile", "Stinger missile", "M4 carbine", "M249 SAW", "M240",
        ],
        "tasks": []
    },
    "military_operations": {
        "templates": ["Describe {seed}", "What happened during {seed}?", "Analyze {seed}"],
        "seeds": [
            "D-Day invasion", "Operation Desert Storm", "Operation Iraqi Freedom",
            "Operation Enduring Freedom", "Battle of Midway", "Battle of the Bulge",
            "Tet Offensive", "Operation Overlord", "Operation Neptune Spear",
            "Operation Red Wings", "Operation Gothic Serpent", "Battle of Fallujah",
            "Siege of Stalingrad", "Battle of Kursk", "Operation Barbarossa",
            "Battle of Britain", "Pearl Harbor attack", "Doolittle Raid",
            "Operation Market Garden", "Inchon Landing", "Battle of Chosin Reservoir",
        ],
        "tasks": []
    },
    "military_ranks": {
        "templates": ["What is {seed}?", "Responsibilities of {seed}", "Insignia of {seed}"],
        "seeds": [
            # Enlisted
            "Private", "Private First Class", "Specialist", "Corporal", "Sergeant",
            "Staff Sergeant", "Sergeant First Class", "Master Sergeant", "First Sergeant",
            "Sergeant Major", "Command Sergeant Major", "Sergeant Major of the Army",
            # Officers
            "Second Lieutenant", "First Lieutenant", "Captain", "Major",
            "Lieutenant Colonel", "Colonel", "Brigadier General", "Major General",
            "Lieutenant General", "General", "General of the Army",
            # Navy
            "Seaman", "Petty Officer", "Chief Petty Officer", "Ensign", "Lieutenant",
            "Commander", "Captain", "Rear Admiral", "Vice Admiral", "Admiral",
        ],
        "tasks": []
    },

    # ─────────────────────────────────────────────────────────────────────────────
    # SPACE & ASTRONOMY
    # ─────────────────────────────────────────────────────────────────────────────
    "planets_moons": {
        "templates": ["Tell me about {seed}", "Facts about {seed}", "What is unique about {seed}?"],
        "seeds": [
            # Planets
            "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
            # Dwarf Planets
            "Pluto", "Ceres", "Eris", "Makemake", "Haumea",
            # Major Moons
            "Moon", "Phobos", "Deimos", "Io", "Europa", "Ganymede", "Callisto",
            "Titan", "Enceladus", "Mimas", "Rhea", "Iapetus", "Triton", "Charon",
            "Miranda", "Ariel", "Umbriel", "Titania", "Oberon",
            # Features
            "Great Red Spot", "Olympus Mons", "Valles Marineris", "Caloris Basin",
            "Saturn's rings", "Jupiter's bands", "Titan's methane lakes",
        ],
        "tasks": []
    },
    "space_missions": {
        "templates": ["Describe {seed}", "What did {seed} accomplish?", "History of {seed}"],
        "seeds": [
            # NASA Missions
            "Apollo 11", "Apollo 13", "Voyager 1", "Voyager 2", "Curiosity rover",
            "Perseverance rover", "Mars Pathfinder", "Spirit rover", "Opportunity rover",
            "Cassini-Huygens", "New Horizons", "Juno", "Parker Solar Probe",
            "James Webb Space Telescope", "Hubble Space Telescope", "Kepler mission",
            "TESS mission", "InSight lander", "OSIRIS-REx", "Lucy mission",
            # Historic
            "Sputnik 1", "Vostok 1", "Mercury program", "Gemini program", "Skylab",
            "Space Shuttle program", "International Space Station",
            # Other Agencies
            "Chang'e missions", "Hayabusa missions", "Rosetta mission", "BepiColombo",
            "ExoMars", "Chandrayaan missions", "Mars Orbiter Mission",
        ],
        "tasks": []
    },
    "space_agencies": {
        "templates": ["Tell me about {seed}", "What does {seed} do?", "History of {seed}"],
        "seeds": [
            "NASA", "ESA", "Roscosmos", "CNSA", "JAXA", "ISRO", "CSA",
            "SpaceX", "Blue Origin", "Virgin Galactic", "Rocket Lab", "Arianespace",
            "United Launch Alliance", "Northrop Grumman", "Boeing Space",
            "Lockheed Martin Space", "Axiom Space", "Sierra Space",
        ],
        "tasks": []
    },
    "astronomical_objects": {
        "templates": ["Explain {seed}", "What is {seed}?", "How does {seed} form?"],
        "seeds": [
            # Stars
            "Sun", "Proxima Centauri", "Alpha Centauri", "Betelgeuse", "Sirius",
            "Polaris", "Vega", "Rigel", "Aldebaran", "Arcturus",
            # Galaxies
            "Milky Way", "Andromeda Galaxy", "Triangulum Galaxy", "Large Magellanic Cloud",
            "Small Magellanic Cloud", "Whirlpool Galaxy", "Sombrero Galaxy",
            # Other Objects
            "Orion Nebula", "Crab Nebula", "Horsehead Nebula", "Pillars of Creation",
            "Sagittarius A*", "Cygnus X-1", "Halley's Comet", "Asteroid belt",
            "Kuiper Belt", "Oort Cloud", "Asteroid Bennu", "Asteroid Ryugu",
        ],
        "tasks": []
    },
    "astronauts_cosmonauts": {
        "templates": ["Who is {seed}?", "Achievements of {seed}", "Tell me about {seed}"],
        "seeds": [
            "Neil Armstrong", "Buzz Aldrin", "Michael Collins", "John Glenn",
            "Alan Shepard", "Yuri Gagarin", "Valentina Tereshkova", "Sally Ride",
            "Mae Jemison", "Chris Hadfield", "Scott Kelly", "Peggy Whitson",
            "Christina Koch", "Jessica Meir", "Sunita Williams", "Kalpana Chawla",
            "Yang Liwei", "Tim Peake", "Samantha Cristoforetti", "Thomas Pesquet",
        ],
        "tasks": []
    },

    # ─────────────────────────────────────────────────────────────────────────────
    # SCIENCE - DETAILED BREAKDOWNS
    # ─────────────────────────────────────────────────────────────────────────────
    "chemical_elements": {
        "templates": ["Properties of {seed}", "Uses of {seed}", "Tell me about {seed}"],
        "seeds": [
            "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen",
            "Oxygen", "Fluorine", "Neon", "Sodium", "Magnesium", "Aluminum", "Silicon",
            "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
            "Iron", "Copper", "Zinc", "Silver", "Gold", "Platinum", "Mercury",
            "Lead", "Uranium", "Plutonium", "Titanium", "Nickel", "Cobalt", "Tungsten",
        ],
        "tasks": []
    },
    "biological_systems": {
        "templates": ["Explain {seed}", "How does {seed} work?", "Components of {seed}"],
        "seeds": [
            "Circulatory system", "Respiratory system", "Digestive system", "Nervous system",
            "Skeletal system", "Muscular system", "Endocrine system", "Immune system",
            "Lymphatic system", "Urinary system", "Reproductive system", "Integumentary system",
            "Cell membrane", "Mitochondria", "Nucleus", "Ribosome", "Golgi apparatus",
            "Endoplasmic reticulum", "Chloroplast", "Vacuole", "Cytoplasm",
        ],
        "tasks": []
    },
    "physics_concepts": {
        "templates": ["Explain {seed}", "What is {seed}?", "Applications of {seed}"],
        "seeds": [
            "Newton's laws", "Conservation of energy", "Conservation of momentum",
            "Thermodynamics laws", "Electromagnetism", "Quantum mechanics",
            "Special relativity", "General relativity", "Wave-particle duality",
            "Heisenberg uncertainty principle", "Schrödinger equation", "Pauli exclusion",
            "Standard Model", "Higgs boson", "Quarks", "Leptons", "Bosons",
            "Strong force", "Weak force", "Electromagnetic force", "Gravity",
            "Dark matter", "Dark energy", "String theory", "Quantum entanglement",
        ],
        "tasks": []
    },

    # ─────────────────────────────────────────────────────────────────────────────
    # TECHNOLOGY - DETAILED
    # ─────────────────────────────────────────────────────────────────────────────
    "programming_concepts": {
        "templates": ["Explain {seed}", "How does {seed} work?", "Example of {seed}"],
        "seeds": [
            "Variables", "Data types", "Functions", "Classes", "Objects", "Inheritance",
            "Polymorphism", "Encapsulation", "Abstraction", "Interfaces", "Generics",
            "Recursion", "Iteration", "Arrays", "Linked lists", "Hash tables",
            "Trees", "Graphs", "Stacks", "Queues", "Heaps", "Binary search",
            "Sorting algorithms", "Big O notation", "Time complexity", "Space complexity",
            "Concurrency", "Parallelism", "Threads", "Processes", "Async/await",
            "Callbacks", "Promises", "Event loop", "Garbage collection", "Memory management",
        ],
        "tasks": []
    },
    "cloud_services": {
        "templates": ["What is {seed}?", "How to use {seed}", "Benefits of {seed}"],
        "seeds": [
            # AWS
            "AWS EC2", "AWS S3", "AWS Lambda", "AWS RDS", "AWS DynamoDB", "AWS EKS",
            "AWS CloudFormation", "AWS IAM", "AWS VPC", "AWS Route 53", "AWS CloudFront",
            # GCP
            "Google Cloud Compute", "Google Cloud Storage", "Cloud Functions",
            "BigQuery", "Cloud Spanner", "GKE", "Cloud Run", "Pub/Sub",
            # Azure
            "Azure VMs", "Azure Blob Storage", "Azure Functions", "Azure SQL",
            "Azure Cosmos DB", "AKS", "Azure DevOps", "Azure Active Directory",
            # General
            "Kubernetes", "Docker", "Terraform", "Ansible", "Jenkins", "GitHub Actions",
        ],
        "tasks": []
    },
    "ai_ml_concepts": {
        "templates": ["Explain {seed}", "How does {seed} work?", "Applications of {seed}"],
        "seeds": [
            "Neural networks", "Deep learning", "Machine learning", "Supervised learning",
            "Unsupervised learning", "Reinforcement learning", "Convolutional neural networks",
            "Recurrent neural networks", "LSTM", "Transformer architecture", "Attention mechanism",
            "GPT", "BERT", "Diffusion models", "GANs", "Autoencoders", "VAE",
            "Gradient descent", "Backpropagation", "Activation functions", "Loss functions",
            "Overfitting", "Regularization", "Dropout", "Batch normalization",
            "Transfer learning", "Fine-tuning", "Embeddings", "Tokenization",
            "Prompt engineering", "RAG", "Vector databases", "LangChain",
        ],
        "tasks": []
    },

    # ─────────────────────────────────────────────────────────────────────────────
    # HISTORY - DETAILED PERIODS
    # ─────────────────────────────────────────────────────────────────────────────
    "ancient_civilizations": {
        "templates": ["Tell me about {seed}", "History of {seed}", "What was {seed} known for?"],
        "seeds": [
            "Ancient Egypt", "Ancient Greece", "Roman Empire", "Persian Empire",
            "Mesopotamia", "Sumerian civilization", "Babylonian Empire", "Assyrian Empire",
            "Ancient China", "Han Dynasty", "Qin Dynasty", "Zhou Dynasty",
            "Maurya Empire", "Gupta Empire", "Indus Valley Civilization",
            "Mayan civilization", "Aztec Empire", "Inca Empire", "Olmec civilization",
            "Phoenicians", "Carthage", "Byzantine Empire", "Ottoman Empire",
        ],
        "tasks": []
    },
    "historical_figures": {
        "templates": ["Who was {seed}?", "Achievements of {seed}", "Tell me about {seed}"],
        "seeds": [
            "Alexander the Great", "Julius Caesar", "Augustus", "Cleopatra",
            "Genghis Khan", "Napoleon Bonaparte", "George Washington", "Abraham Lincoln",
            "Winston Churchill", "Franklin D. Roosevelt", "Mahatma Gandhi", "Martin Luther King Jr.",
            "Nelson Mandela", "Albert Einstein", "Isaac Newton", "Galileo Galilei",
            "Leonardo da Vinci", "Michelangelo", "Shakespeare", "Mozart", "Beethoven",
            "Marie Curie", "Charles Darwin", "Nikola Tesla", "Thomas Edison",
        ],
        "tasks": []
    },

    # ─────────────────────────────────────────────────────────────────────────────
    # SPORTS - DETAILED
    # ─────────────────────────────────────────────────────────────────────────────
    "sports_leagues": {
        "templates": ["Tell me about {seed}", "History of {seed}", "Teams in {seed}"],
        "seeds": [
            "NFL", "NBA", "MLB", "NHL", "MLS", "Premier League", "La Liga",
            "Bundesliga", "Serie A", "Ligue 1", "UEFA Champions League", "FIFA World Cup",
            "Olympics", "NCAA", "PGA Tour", "ATP Tour", "WTA Tour", "UFC",
            "WWE", "Formula 1", "NASCAR", "IndyCar", "Tour de France",
        ],
        "tasks": []
    },
    "sports_positions": {
        "templates": ["What does {seed} do?", "Role of {seed}", "Skills needed for {seed}"],
        "seeds": [
            # Football
            "Quarterback", "Running back", "Wide receiver", "Tight end", "Offensive lineman",
            "Linebacker", "Defensive end", "Cornerback", "Safety",
            # Basketball
            "Point guard", "Shooting guard", "Small forward", "Power forward", "Center",
            # Baseball
            "Pitcher", "Catcher", "First baseman", "Shortstop", "Outfielder",
            # Soccer
            "Goalkeeper", "Defender", "Midfielder", "Forward", "Striker", "Winger",
        ],
        "tasks": []
    },

    # ─────────────────────────────────────────────────────────────────────────────
    # ENTERTAINMENT & MEDIA
    # ─────────────────────────────────────────────────────────────────────────────
    "film_genres": {
        "templates": ["What defines {seed}?", "Examples of {seed}", "History of {seed}"],
        "seeds": [
            "Action films", "Comedy films", "Drama", "Horror", "Science fiction",
            "Fantasy", "Thriller", "Romance", "Documentary", "Animation",
            "Western", "Musical", "War films", "Crime films", "Mystery",
            "Adventure", "Biographical films", "Historical films", "Noir",
        ],
        "tasks": []
    },
    "music_genres": {
        "templates": ["What is {seed}?", "History of {seed}", "Artists in {seed}"],
        "seeds": [
            "Rock", "Pop", "Hip hop", "R&B", "Jazz", "Blues", "Country",
            "Classical", "Electronic", "Reggae", "Metal", "Punk", "Folk",
            "Soul", "Funk", "Disco", "House", "Techno", "Dubstep", "Indie",
            "Alternative", "Grunge", "Progressive rock", "Psychedelic rock",
        ],
        "tasks": []
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# DYNAMIC CATEGORY MANAGER
# ════════════════════════════════════════════════════════════════════════════════

class DynamicCategoryManager:
    """Manages dynamic category discovery and expansion."""

    def __init__(self):
        self.categories: Dict[str, DynamicCategory] = {}
        self.extractor = TrainingDataExtractor()
        self.expander = LLMExpander()
        self.stats = DiscoveryStats()

        # Load existing dynamic categories
        self._load_existing()

        # Initialize with knowledge base
        self._init_from_knowledge_base()

    def _load_existing(self):
        """Load existing dynamic categories."""
        if DYNAMIC_CATEGORIES_FILE.exists():
            with open(DYNAMIC_CATEGORIES_FILE) as f:
                data = json.load(f)
                for name, cat_data in data.items():
                    self.categories[name] = DynamicCategory(**cat_data)

    def _init_from_knowledge_base(self):
        """Initialize categories from pre-computed knowledge base."""
        for name, data in KNOWLEDGE_BASE.items():
            if name not in self.categories:
                self.categories[name] = DynamicCategory(
                    name=name,
                    templates=data.get("templates", []),
                    seeds=data.get("seeds", []),
                    tasks=data.get("tasks", []),
                    source="knowledge_base",
                )
                self.stats.categories_discovered += 1
                self.stats.seeds_discovered += len(data.get("seeds", []))

    def discover_from_training_data(self):
        """Discover new seeds from existing training data."""
        print("\n[DISCOVERY] Extracting from training data...")

        # Check all JSONL files in data_store
        jsonl_files = list(DATA_STORE.glob("**/*.jsonl"))

        all_entities = defaultdict(set)
        for filepath in jsonl_files:
            entities = self.extractor.extract_from_jsonl(filepath)
            for category, seeds in entities.items():
                all_entities[category].update(seeds)

        # Add discovered seeds to existing categories or create new ones
        for category, seeds in all_entities.items():
            if category in self.categories:
                existing = set(self.categories[category].seeds)
                new_seeds = seeds - existing
                if new_seeds:
                    self.categories[category].seeds.extend(list(new_seeds))
                    self.stats.seeds_discovered += len(new_seeds)
                    self.stats.from_training_data += len(new_seeds)
            else:
                self.categories[category] = DynamicCategory(
                    name=category,
                    seeds=list(seeds),
                    source="extracted",
                )
                self.stats.categories_discovered += 1
                self.stats.seeds_discovered += len(seeds)
                self.stats.from_training_data += len(seeds)

        print(f"  Discovered {self.stats.from_training_data} seeds from training data")

    async def expand_domain_llm(self, domain: str, depth: int = 2):
        """Expand a domain using LLM with hierarchical depth."""
        print(f"\n[EXPANSION] Expanding domain: {domain}")

        # Get initial expansion
        category = await self.expander.expand_domain(domain)
        if not category:
            return

        self.categories[domain] = category
        self.stats.categories_discovered += 1
        self.stats.seeds_discovered += len(category.seeds)
        self.stats.from_llm_expansion += len(category.seeds)

        print(f"  Found {len(category.seeds)} seeds, {len(category.subcategories)} subcategories")

        # Expand subcategories if depth allows
        if depth > 1 and category.subcategories:
            for subcat in category.subcategories[:10]:  # Limit subcategories
                subcat_name = f"{domain}_{subcat.lower().replace(' ', '_')}"
                seeds = await self.expander.expand_hierarchical(subcat, domain)
                if seeds:
                    self.categories[subcat_name] = DynamicCategory(
                        name=subcat_name,
                        parent=domain,
                        seeds=seeds,
                        source="llm_hierarchical",
                    )
                    self.stats.categories_discovered += 1
                    self.stats.seeds_discovered += len(seeds)
                    self.stats.from_hierarchical += len(seeds)
                    print(f"    Subcategory {subcat}: {len(seeds)} seeds")

                await asyncio.sleep(0.5)  # Rate limiting

    async def auto_expand_all(self, domains: List[str] = None):
        """Automatically expand all root domains."""
        domains = domains or ROOT_DOMAINS

        print(f"\n[AUTO-EXPAND] Processing {len(domains)} domains...")

        for domain in domains:
            # Skip if already expanded with enough seeds
            if domain in self.categories and len(self.categories[domain].seeds) > 20:
                print(f"  Skipping {domain} (already has {len(self.categories[domain].seeds)} seeds)")
                continue

            await self.expand_domain_llm(domain, depth=2)
            await asyncio.sleep(1)  # Rate limiting between domains

    def save(self):
        """Save dynamic categories to file."""
        DATA_STORE.mkdir(parents=True, exist_ok=True)

        data = {}
        for name, cat in self.categories.items():
            data[name] = asdict(cat)

        with open(DYNAMIC_CATEGORIES_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[SAVED] {len(self.categories)} categories to {DYNAMIC_CATEGORIES_FILE}")

    def export_to_domains_format(self) -> Dict:
        """Export all categories in DOMAINS format for generate_all_training_data.py."""
        domains = {}
        for name, cat in self.categories.items():
            if cat.seeds:  # Only export categories with seeds
                domains[name] = cat.to_domain_format()
        return domains

    def merge_into_generator(self):
        """Merge dynamic categories into generate_all_training_data.py's DOMAINS."""
        from generate_all_training_data import DOMAINS, CATEGORIES

        merged = 0
        for name, cat in self.categories.items():
            if name not in DOMAINS and cat.seeds:
                DOMAINS[name] = cat.to_domain_format()
                CATEGORIES[name] = {
                    "weight": 1.0,
                    "description": f"Dynamic: {name}",
                    "templates": cat.templates or cat._default_templates(),
                    "seeds": cat.seeds,
                    "tasks": cat.tasks,
                }
                merged += 1

        print(f"[MERGED] {merged} dynamic categories into generator")
        return merged

    def print_stats(self):
        """Print discovery statistics."""
        tracker = get_seed_tracker()

        print(f"\n{'═' * 60}")
        print("DYNAMIC CATEGORY STATISTICS")
        print(f"{'═' * 60}")
        print(f"  Total categories: {len(self.categories)}")
        total_seeds = sum(len(c.seeds) for c in self.categories.values())
        print(f"  Total seeds: {total_seeds}")
        print(f"  From knowledge base: {len(KNOWLEDGE_BASE)}")
        print(f"  From training data: {self.stats.from_training_data}")
        print(f"  From LLM expansion: {self.stats.from_llm_expansion}")
        print(f"  From hierarchical: {self.stats.from_hierarchical}")

        # Seed usage statistics
        if tracker:
            tracker_stats = tracker.get_stats()
            total_used = tracker_stats['total_unique_seeds']
            total_uses = tracker_stats['total_uses']
            coverage_pct = (total_used / total_seeds * 100) if total_seeds > 0 else 0
            print(f"\n  SEED USAGE TRACKING:")
            print(f"    Seeds used: {total_used}/{total_seeds} ({coverage_pct:.1f}% coverage)")
            print(f"    Total uses: {total_uses}")
            print(f"    Unused seeds remaining: {total_seeds - total_used}")

        # Top categories by seed count
        print(f"\n  Top categories by seeds:")
        sorted_cats = sorted(self.categories.items(), key=lambda x: len(x[1].seeds), reverse=True)
        for name, cat in sorted_cats[:15]:
            # Show coverage for each category
            if tracker:
                used, total, pct = tracker.get_coverage(cat.seeds, name)
                status = f"[{used}/{total} used]"
            else:
                status = ""
            print(f"    {name:<30} {len(cat.seeds):>5} seeds {status}")

        # Show categories with lowest coverage (most opportunity)
        if tracker:
            print(f"\n  Categories with most unused seeds:")
            cat_coverage = []
            for name, cat in self.categories.items():
                if cat.seeds:
                    unused = tracker.get_unused_seeds(cat.seeds, name)
                    cat_coverage.append((name, len(unused), len(cat.seeds)))
            cat_coverage.sort(key=lambda x: x[1], reverse=True)
            for name, unused, total in cat_coverage[:10]:
                print(f"    {name:<30} {unused:>5}/{total} unused")


# ════════════════════════════════════════════════════════════════════════════════
# AUTO-INJECT INTO GENERATOR
# ════════════════════════════════════════════════════════════════════════════════

def inject_dynamic_categories():
    """
    Called by generate_all_training_data.py to inject dynamic categories.
    Returns dict of additional DOMAINS entries.
    """
    manager = DynamicCategoryManager()
    return manager.export_to_domains_format()


def get_all_categories() -> Dict:
    """
    Get all categories (static + dynamic) merged together.
    Use this instead of DOMAINS directly for maximum coverage.
    """
    # Import static categories
    from generate_all_training_data import DOMAINS

    # Get dynamic categories
    manager = DynamicCategoryManager()
    dynamic = manager.export_to_domains_format()

    # Merge (dynamic overwrites static if conflict)
    merged = {**DOMAINS, **dynamic}
    return merged


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Dynamic Category Discovery & Expansion")
    parser.add_argument("--discover", action="store_true", help="Discover from training data")
    parser.add_argument("--expand", action="store_true", help="LLM-expand all root domains")
    parser.add_argument("--expand-domain", type=str, help="Expand specific domain")
    parser.add_argument("--auto", action="store_true", help="Full auto-discovery pipeline")
    parser.add_argument("--merge", action="store_true", help="Merge into generator")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--export", type=str, help="Export to file")
    args = parser.parse_args()

    manager = DynamicCategoryManager()

    if args.discover or args.auto:
        manager.discover_from_training_data()

    if args.expand or args.auto:
        await manager.auto_expand_all()

    if args.expand_domain:
        await manager.expand_domain_llm(args.expand_domain, depth=2)

    if args.merge:
        manager.merge_into_generator()

    if args.stats or args.auto:
        manager.print_stats()

    if args.export:
        domains = manager.export_to_domains_format()
        with open(args.export, 'w') as f:
            json.dump(domains, f, indent=2)
        print(f"Exported to {args.export}")

    # Always save
    manager.save()


if __name__ == "__main__":
    asyncio.run(main())
