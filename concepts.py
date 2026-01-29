"""
Foundational Dictionary of Concepts
===================================
This module provides a bridge between raw random seeds and sophisticated
conceptual starting points for reasoning chain generation.

It ensures that the "pseudo random seed" deterministically maps to
specific nodes in the knowledge manifold, forcing the model to
reason about diverse and concrete topics.
"""

from typing import List, Optional
import random
import json
import os

class FoundationalDictionary:
    """
    Manages the foundational concepts and their deterministic selection.
    """
    
    # A starter set of high-level concepts across various domains.
    # In a full production system, this could be loaded from a large corpus or WordNet.
    CONCEPTS = [
        # Abstract / Philosophical
        "Entropy", "Recursion", "Determinism", "Epistemology", "Paradox", "Ontology", 
        "Utilitarianism", "Existentialism", "Dualism", "Causality", "Abstraction",
        "Metaphor", "Allegory", "Dialectic", "Axiom", "Heuristic",
        
        # Science / Physics / Math
        "Thermodynamics", "Quantum Superposition", "Relativity", "Evolution", "Homeostasis",
        "Photosynthesis", "Tectonics", "Neuroplasticity", "Algorithm", "Cryptography",
        "Topology", "Fractal", "Vector", "Probability", "Isomorphism", "Symmetry",
        
        # Social / Political / Economic
        "Democracy", "Capitalism", "Socialism", "Jurisprudence", "Inflation", "Globalization",
        "Bureaucracy", "Meritocracy", "Diplomacy", "Sovereignty", "Equity", "Liberty",
        "Propaganda", "Urbanization", "Demographics", "Game Theory",
        
        # Arts / Culture
        "Surrealism", "Minimalism", "Renaissance", "Post-modernism", "Narrative", "Archetype",
        "Harmony", "Dissonance", "Perspective", "Symbolism", "Genre", "Canon",
        
        # Technology
        "Artificial Intelligence", "Blockchain", "Virtual Reality", "Automation", "Interface",
        "Network", "Latency", "Bandwidth", "Protocol", "Cybersecurity", "Singularity",
        
        # Biological / Psychological
        "Cognition", "Perception", "Consciousness", "Instinct", "Memory", "Trauma",
        "Resilience", "Adaptation", "Symbiosis", "Parasitism", "Mutation", "Metabolism"
        "Resilience", "Adaptation", "Symbiosis", "Parasitism", "Mutation", "Metabolism"
    ]

    def __init__(self, concepts_file: str = "optional_unverified_concepts/foundational_concepts.json"):
        # Start with static concepts
        concept_set = set(self.CONCEPTS)
        
        # Try to load dynamic concepts
        if os.path.exists(concepts_file):
            try:
                with open(concepts_file, 'r') as f:
                    dynamic = json.load(f)
                    if isinstance(dynamic, list):
                        concept_set.update(dynamic)
                        print(f"  [Concept Bridge] Loaded {len(dynamic)} concepts from {concepts_file}")
                    elif isinstance(dynamic, dict):
                        concept_set.update(dynamic.keys())
                        print(f"  [Concept Bridge] Loaded {len(dynamic)} concepts from {concepts_file}")
            except Exception as e:
                print(f"  [Concept Bridge] Failed to load {concepts_file}: {e}")
                
        self._concepts = sorted(list(concept_set)) # Sort for stability

    def get_concepts_from_seed(self, seed: int, count: int = 3) -> List[str]:
        """
        Deterministically selects a list of concepts based on the provided seed.
        
        Args:
            seed: The integer seed to drive the random selection.
            count: Number of concepts to select.
            
        Returns:
            A list of concept strings.
        """
        # Create a private random instance to avoid side effects on global state
        rng = random.Random(seed)
        
        # Select 'count' unique concepts
        # We use sample to get unique items
        if count > len(self._concepts):
            count = len(self._concepts)
            
        return rng.sample(self._concepts, count)

    def get_concept_combo_string(self, seed: int, count: int = 3) -> str:
        """
        Returns a formatted string of concepts suitable for prompt injection.
        """
        concepts = self.get_concepts_from_seed(seed, count)
        return ", ".join(concepts)
