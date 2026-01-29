import random
import re
from collections import deque
from dataclasses import dataclass
from typing import Optional

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, word_tokenize
    HAS_NLTK = True
except Exception:
    nltk = None
    wn = None
    WordNetLemmatizer = None
    pos_tag = None
    word_tokenize = None
    HAS_NLTK = False


def ensure_nltk_resource(path: str, name: str) -> bool:
    if not HAS_NLTK:
        return False
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        try:
            nltk.download(name, quiet=True)
            nltk.data.find(path)
            return True
        except Exception:
            return False


HAS_PUNKT = ensure_nltk_resource("tokenizers/punkt", "punkt")
HAS_TAGGER = ensure_nltk_resource(
    "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger",
)
HAS_WORDNET = ensure_nltk_resource("corpora/wordnet", "wordnet")
_ = ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")

lemmatizer = WordNetLemmatizer() if HAS_WORDNET else None

STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "in",
    "on",
    "for",
    "with",
    "and",
    "or",
    "of",
    "at",
    "by",
    "from",
    "your",
    "their",
    "our",
    "into",
    "over",
    "under",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    if HAS_PUNKT:
        return [t for t in word_tokenize(text) if re.search(r"[a-z]", t)]
    return re.findall(r"[a-z]+", text)


def pos_tag_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    if HAS_TAGGER:
        return pos_tag(tokens)
    return [(t, "NN") for t in tokens]


def penn_to_wn(tag: str):
    if not HAS_WORDNET:
        return None
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("R"):
        return wn.ADV
    return None


def simple_lemma(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def normalize_phrase(text: str) -> set[str]:
    tokens = tokenize(text)
    if not tokens:
        return set()
    if HAS_WORDNET and HAS_TAGGER and lemmatizer:
        tagged = pos_tag_tokens(tokens)
        lemmas = set()
        for tok, tag in tagged:
            wn_pos = penn_to_wn(tag)
            if wn_pos is None:
                continue
            lemmas.add(lemmatizer.lemmatize(tok, wn_pos))
        return lemmas
    return {simple_lemma(tok) for tok in tokens if tok not in STOPWORDS}


def expand_seed_terms(seeds: list[str], max_depth: int = 1) -> set[str]:
    expanded = set()

    def add_from_synset(synset, depth: int):
        for lemma in synset.lemmas():
            expanded.add(lemma.name().lower().replace("_", " "))

        if depth <= 0:
            return

        for rel in synset.hypernyms() + synset.hyponyms():
            add_from_synset(rel, depth - 1)

    for seed in seeds:
        for token in tokenize(seed):
            if HAS_WORDNET:
                for syn in wn.synsets(token, pos=wn.VERB) + wn.synsets(
                    token,
                    pos=wn.NOUN,
                ):
                    add_from_synset(syn, max_depth)
            else:
                expanded.add(token)

    normalized_tokens = set()
    for phrase in expanded:
        for t in tokenize(phrase):
            if t not in STOPWORDS:
                normalized_tokens.add(t)

    return normalized_tokens


def semantic_signature(lemmas: set[str], max_depth: int = 1) -> set[str]:
    signature = set()
    for lemma in lemmas:
        if lemma in STOPWORDS:
            continue
        signature.add(lemma)
        if not HAS_WORDNET:
            continue
        for syn in wn.synsets(lemma, pos=wn.VERB) + wn.synsets(lemma, pos=wn.NOUN):
            for l in syn.lemmas():
                for token in tokenize(l.name().replace("_", " ")):
                    if token not in STOPWORDS:
                        signature.add(token)
            if max_depth <= 0:
                continue
            for rel in syn.hypernyms() + syn.hyponyms():
                for l in rel.lemmas():
                    for token in tokenize(l.name().replace("_", " ")):
                        if token not in STOPWORDS:
                            signature.add(token)
    return signature


VERB_SEEDS = [
    "volunteer",
    "donate",
    "recycle",
    "help",
    "mentor",
    "rescue",
    "assist",
    "cooperate",
    "educate",
    "foster",
    "report",
    "clean",
    "care",
    "support",
    "protect",
    "advocate",
    "organize",
    "tutor",
    "teach",
    "plant",
    "restore",
    "conserve",
    "repair",
    "refurbish",
    "translate",
    "share",
    "create",
    "host",
    "participate",
    "attend",
    "register",
    "serve",
    "deliver",
    "distribute",
    "provide",
    "offer",
    "install",
    "assemble",
    "check",
    "reduce",
    "build",
    "collect",
    "stock",
    "weatherize",
    "fundraise",
    "coach",
    "prepare",
    "cook",
    "maintain",
    "equip",
    "coordinate",
    "train",
    "steward",
    "mobilize",
    "empower",
    "facilitate",
    "partner",
]

NOUN_SEEDS = [
    "community",
    "neighborhood",
    "neighbors",
    "shelter",
    "food bank",
    "food pantry",
    "food",
    "pantry",
    "meal",
    "groceries",
    "public",
    "school",
    "clinic",
    "hospital",
    "park",
    "library",
    "seniors",
    "students",
    "children",
    "animals",
    "wildlife",
    "accessibility",
    "disaster",
    "relief",
    "health",
    "safety",
    "housing",
    "homelessness",
    "homeless",
    "outreach",
    "hygiene",
    "blanket",
    "coat",
    "clothing",
    "goods",
    "blood",
    "platelet",
    "medical",
    "vaccination",
    "environment",
    "water",
    "air",
    "emission",
    "transit",
    "wheelchair",
    "ramp",
    "pollinator",
    "literacy",
    "business",
    "device",
    "laptop",
    "election",
    "voter",
    "governance",
    "families",
    "parents",
    "caregivers",
    "youth",
    "teens",
    "veterans",
    "immigrants",
    "refugees",
    "newcomers",
    "community garden",
    "garden",
    "trails",
    "beach",
    "river",
    "stream",
    "lake",
    "bike lane",
    "sidewalk",
    "transit",
    "bus stop",
    "energy",
    "efficiency",
    "weatherization",
    "solar",
    "arts",
    "music",
    "culture",
    "childcare",
    "after school",
    "job training",
    "resume",
    "interview",
    "nutrition",
    "farmers market",
    "wellness",
]

ACTION_VERBS = set()
for verb in VERB_SEEDS:
    ACTION_VERBS.update(tokenize(verb))

BENEFICIAL_VOCAB = expand_seed_terms(VERB_SEEDS + NOUN_SEEDS, max_depth=1)
BENEFICIARY_VOCAB = expand_seed_terms(NOUN_SEEDS, max_depth=1)

DIRECT_SERVICE_VERBS = set()
for verb in [
    "volunteer",
    "donate",
    "deliver",
    "distribute",
    "serve",
    "assemble",
    "install",
    "tutor",
    "mentor",
    "teach",
    "clean",
    "plant",
    "restore",
    "repair",
    "refurbish",
    "translate",
    "create",
    "build",
    "collect",
    "stock",
    "provide",
    "offer",
    "foster",
    "rescue",
    "register",
    "report",
    "help",
    "coach",
    "prepare",
    "cook",
    "maintain",
    "equip",
    "train",
    "weatherize",
]:
    DIRECT_SERVICE_VERBS.update(tokenize(verb))

INDIRECT_VERBS = set()
for verb in ["support", "advocate", "attend", "participate"]:
    INDIRECT_VERBS.update(tokenize(verb))

IMPACT_PHRASES = [
    "food bank",
    "food pantry",
    "community fridge",
    "meals",
    "groceries",
    "shelter",
    "housing",
    "homelessness",
    "emergency",
    "disaster",
    "storm",
    "heat wave",
    "outage",
    "clinic",
    "hospital",
    "vaccination",
    "blood",
    "platelets",
    "medical supplies",
    "hygiene kits",
    "mental health",
    "substance recovery",
    "school",
    "literacy",
    "voter registration",
    "election",
    "public transit",
    "accessibility",
    "wheelchair ramp",
    "carbon monoxide",
    "smoke detector",
    "crosswalk",
    "water quality",
    "air quality",
    "wildfire",
    "pollinator",
    "river",
    "wetlands",
    "food waste",
    "emissions",
    "community garden",
    "after school programs",
    "youth mentoring",
    "job training",
    "resume review",
    "bike lanes",
    "safe sidewalks",
    "transit stops",
    "bus shelters",
    "energy efficiency",
    "weatherization",
    "solar access",
    "cooling centers",
    "heat relief",
    "arts programs",
    "public art",
    "music education",
    "trail cleanup",
    "beach cleanup",
    "tree canopy",
]
IMPACT_TOKENS = set()
for phrase in IMPACT_PHRASES:
    IMPACT_TOKENS.update(tokenize(phrase))


def score_from_lemmas(lemmas: set[str]) -> float:
    if not lemmas or not (lemmas & ACTION_VERBS):
        return 0.0
    if len(lemmas) < 2:
        return 0.0
    if not (lemmas & BENEFICIARY_VOCAB or lemmas & IMPACT_TOKENS):
        return 0.0
    beneficial_overlap = len(lemmas & BENEFICIAL_VOCAB)
    beneficiary_overlap = len(lemmas & BENEFICIARY_VOCAB)
    impact_overlap = len(lemmas & IMPACT_TOKENS)
    base = (beneficial_overlap / len(lemmas)) + 0.25 * (
        beneficiary_overlap / len(lemmas)
    )
    direct_bonus = 0.12 if lemmas & DIRECT_SERVICE_VERBS else 0.0
    indirect_penalty = (
        0.08 if (lemmas & INDIRECT_VERBS and not lemmas & DIRECT_SERVICE_VERBS) else 0.0
    )
    impact_score = 0.2 * (impact_overlap / len(lemmas))
    return base + impact_score + direct_bonus - indirect_penalty


def score_behavior(behavior_text: str) -> float:
    return score_from_lemmas(normalize_phrase(behavior_text))


def is_societally_beneficial(behavior_text: str, threshold: float = 0.3) -> bool:
    return score_behavior(behavior_text) >= threshold


SEED_BEHAVIORS = [
    "Donate unused clothing and goods to shelters",
    "Help elderly neighbors with groceries",
    "Volunteer at a local food pantry",
    "Plant native trees in community parks",
    "Teach digital literacy to seniors",
    "Organize a neighborhood cleanup",
    "Deliver meals to homebound neighbors",
    "Build a wheelchair ramp for a neighbor",
]

CAUSE_TEMPLATES = {
    "accessibility": [
        {
            "template": "Advocate for {object}",
            "fills": [
                {"object": "accessible public spaces"},
                {"object": "captioned public events"},
                {"object": "inclusive playgrounds"},
            ],
        },
        {
            "template": "Create {object}",
            "fills": [
                {"object": "large-print materials for community services"},
                {"object": "clear signage for public buildings"},
            ],
        },
        {
            "template": "Help {people} {task}",
            "fills": [
                {"people": "neighbors with disabilities", "task": "get to appointments"},
                {"people": "seniors", "task": "complete online service forms"},
                {"people": "parents with strollers", "task": "navigate public spaces"},
            ],
        },
        {
            "template": "Organize {event}",
            "fills": [
                {"event": "an accessibility audit walk"},
                {"event": "a sign-language practice group"},
            ],
        },
        {
            "template": "Check {object} for accessibility issues",
            "fills": [
                {"object": "websites"},
                {"object": "community event venues"},
                {"object": "public transit routes"},
            ],
        },
    ],
    "animal": [
        {
            "template": "Foster {object}",
            "fills": [
                {"object": "rescue animals"},
                {"object": "kittens and puppies"},
            ],
        },
        {
            "template": "Volunteer at {object}",
            "fills": [
                {"object": "animal shelters"},
                {"object": "wildlife rehabilitation centers"},
            ],
        },
        {
            "template": "Organize {object}",
            "fills": [
                {"object": "a pet food drive"},
                {"object": "a low-cost spay/neuter clinic fundraiser"},
            ],
        },
        {
            "template": "Report {object}",
            "fills": [
                {"object": "lost pets"},
                {"object": "animal cruelty concerns"},
            ],
        },
        {
            "template": "Build {object}",
            "fills": [
                {"object": "shelters for stray cats"},
                {"object": "birdhouses for local wildlife"},
            ],
        },
    ],
    "civic": [
        {
            "template": "Register {people}",
            "fills": [
                {"people": "voters at community events"},
                {"people": "new residents for local elections"},
            ],
        },
        {
            "template": "Serve as {object}",
            "fills": [
                {"object": "an election poll worker"},
                {"object": "a community board volunteer"},
            ],
        },
        {
            "template": "Attend {object}",
            "fills": [
                {"object": "public town hall meetings"},
                {"object": "school board meetings"},
                {"object": "public budget hearings"},
            ],
        },
        {
            "template": "Advocate for {object}",
            "fills": [
                {"object": "transparent local governance"},
                {"object": "fair housing policies"},
            ],
        },
        {
            "template": "Translate {object}",
            "fills": [
                {"object": "public notices for non-English speakers"},
                {"object": "health information for immigrant communities"},
            ],
        },
    ],
    "community": [
        {
            "template": "Volunteer at {object}",
            "fills": [
                {"object": "local shelters"},
                {"object": "community centers"},
                {"object": "libraries"},
            ],
        },
        {
            "template": "Organize {object}",
            "fills": [
                {"object": "a neighborhood cleanup"},
                {"object": "a mutual aid network"},
                {"object": "a community potluck"},
            ],
        },
        {
            "template": "Check in on {people}",
            "fills": [
                {"people": "elderly neighbors"},
                {"people": "new parents"},
                {"people": "neighbors during heat waves"},
            ],
        },
        {
            "template": "Share {object}",
            "fills": [
                {"object": "excess garden produce"},
                {"object": "tools through a lending library"},
                {"object": "rides to appointments"},
            ],
        },
        {
            "template": "Support {object}",
            "fills": [
                {"object": "community art events"},
                {"object": "public parks"},
            ],
        },
        {
            "template": "Coordinate {object}",
            "fills": [
                {"object": "a community mutual aid schedule"},
                {"object": "a community volunteer sign-up rotation"},
                {"object": "a community ride share"},
            ],
        },
    ],
    "food": [
        {
            "template": "Volunteer at {object}",
            "fills": [
                {"object": "food banks"},
                {"object": "food pantries"},
            ],
        },
        {
            "template": "Serve {object}",
            "fills": [
                {"object": "meals at a shelter"},
                {"object": "meals at community kitchens"},
            ],
        },
        {
            "template": "Stock {object}",
            "fills": [
                {"object": "community fridges"},
                {"object": "food pantry shelves"},
            ],
        },
        {
            "template": "Deliver {object}",
            "fills": [
                {"object": "groceries to homebound neighbors"},
                {"object": "meals to seniors"},
            ],
        },
        {
            "template": "Organize {object}",
            "fills": [
                {"object": "a neighborhood food drive"},
                {"object": "a community meal service"},
            ],
        },
        {
            "template": "Prepare {object}",
            "fills": [
                {"object": "meal kits for families"},
                {"object": "snack packs for students"},
                {"object": "fresh meals for neighbors"},
            ],
        },
    ],
    "housing": [
        {
            "template": "Assemble {object}",
            "fills": [
                {"object": "hygiene kits for shelters"},
                {"object": "winter care packages for neighbors experiencing homelessness"},
            ],
        },
        {
            "template": "Volunteer with {object}",
            "fills": [
                {"object": "homelessness outreach teams"},
                {"object": "housing repair nonprofits"},
            ],
        },
        {
            "template": "Help {people} {task}",
            "fills": [
                {"people": "seniors", "task": "weatherize their homes"},
                {"people": "low-income neighbors", "task": "complete housing assistance applications"},
            ],
        },
        {
            "template": "Donate {object}",
            "fills": [
                {"object": "blankets to shelters"},
                {"object": "winter coats to outreach programs"},
            ],
        },
        {
            "template": "Build {object}",
            "fills": [
                {"object": "wheelchair ramps for neighbors in need"},
            ],
        },
        {
            "template": "Repair {object}",
            "fills": [
                {"object": "home safety hazards for seniors"},
                {"object": "broken steps for neighbors"},
                {"object": "leaky fixtures in affordable housing"},
            ],
        },
    ],
    "digital": [
        {
            "template": "Refurbish {object}",
            "fills": [
                {"object": "old laptops for reuse"},
                {"object": "phones for emergency use"},
            ],
        },
        {
            "template": "Teach {subject} to {people}",
            "fills": [
                {"subject": "online safety", "people": "teens"},
                {"subject": "basic smartphone skills", "people": "seniors"},
                {"subject": "remote job tools", "people": "community members"},
            ],
        },
        {
            "template": "Offer {object}",
            "fills": [
                {"object": "free Wi-Fi in community spaces"},
                {"object": "public computer lab hours"},
            ],
        },
        {
            "template": "Translate {object}",
            "fills": [
                {"object": "digital guides into multiple languages"},
                {"object": "online resources for newcomers"},
            ],
        },
        {
            "template": "Create {object}",
            "fills": [
                {"object": "step-by-step guides for accessing public services"},
                {"object": "simple how-to videos for telehealth"},
            ],
        },
        {
            "template": "Train {people} on {subject}",
            "fills": [
                {"people": "community members", "subject": "basic cybersecurity"},
                {"people": "seniors", "subject": "telehealth tools"},
                {"people": "job seekers", "subject": "online applications"},
            ],
        },
    ],
    "disaster": [
        {
            "template": "Assemble {object}",
            "fills": [
                {"object": "emergency kits for neighbors"},
                {"object": "care packages for shelters"},
            ],
        },
        {
            "template": "Volunteer with {object}",
            "fills": [
                {"object": "disaster relief organizations"},
                {"object": "community emergency response teams"},
            ],
        },
        {
            "template": "Collect {object}",
            "fills": [
                {"object": "relief supplies after disasters"},
                {"object": "blankets for emergency shelters"},
            ],
        },
        {
            "template": "Provide {object}",
            "fills": [
                {"object": "charging access during outages"},
                {"object": "information about local shelters"},
            ],
        },
        {
            "template": "Check on {people}",
            "fills": [
                {"people": "neighbors during storms"},
                {"people": "vulnerable residents during heat waves"},
            ],
        },
    ],
    "economic": [
        {
            "template": "Support {object}",
            "fills": [
                {"object": "local small businesses"},
                {"object": "worker-owned businesses"},
                {"object": "fair-trade goods"},
            ],
        },
        {
            "template": "Mentor {people}",
            "fills": [
                {"people": "job seekers"},
                {"people": "first-time entrepreneurs"},
                {"people": "students exploring careers"},
            ],
        },
        {
            "template": "Organize {object}",
            "fills": [
                {"object": "a community job skills workshop"},
                {"object": "a community resume review clinic"},
                {"object": "a community job fair"},
            ],
        },
        {
            "template": "Donate {object}",
            "fills": [
                {"object": "professional clothing to job seekers"},
                {"object": "tools to trade schools"},
            ],
        },
    ],
    "education": [
        {
            "template": "Tutor {people} in {subject}",
            "fills": [
                {"people": "students", "subject": "math or reading"},
                {"people": "English learners", "subject": "conversation practice"},
                {"people": "adults", "subject": "basic literacy"},
            ],
        },
        {
            "template": "Mentor {people}",
            "fills": [
                {"people": "a student"},
                {"people": "a first-generation college applicant"},
                {"people": "a new teacher"},
            ],
        },
        {
            "template": "Donate {object}",
            "fills": [
                {"object": "books to schools"},
                {"object": "school supplies to classrooms"},
                {"object": "used laptops to learning centers"},
            ],
        },
        {
            "template": "Teach {subject} to {people}",
            "fills": [
                {"subject": "digital literacy", "people": "seniors"},
                {"subject": "coding basics", "people": "students"},
                {"subject": "financial literacy", "people": "community members"},
            ],
        },
        {
            "template": "Host {event}",
            "fills": [
                {"event": "a free study group"},
                {"event": "a community science night"},
                {"event": "a resume workshop"},
            ],
        },
    ],
    "environment": [
        {
            "template": "Clean up {object}",
            "fills": [
                {"object": "litter in public spaces"},
                {"object": "community parks"},
                {"object": "riverbanks"},
                {"object": "public trails"},
            ],
        },
        {
            "template": "Plant {object}",
            "fills": [
                {"object": "native trees"},
                {"object": "pollinator gardens"},
                {"object": "native shrubs"},
            ],
        },
        {
            "template": "Restore {object}",
            "fills": [
                {"object": "stream banks"},
                {"object": "wetlands"},
                {"object": "local habitats"},
            ],
        },
        {
            "template": "Reduce {object}",
            "fills": [
                {"object": "food waste"},
                {"object": "single-use plastics"},
                {"object": "household energy use"},
            ],
        },
        {
            "template": "Organize a cleanup for {object}",
            "fills": [
                {"object": "beaches"},
                {"object": "parks"},
                {"object": "neighborhood streets"},
            ],
        },
        {
            "template": "Protect {object}",
            "fills": [
                {"object": "local ecosystems"},
                {"object": "water quality"},
                {"object": "urban tree canopy"},
            ],
        },
        {
            "template": "{action}",
            "fills": [
                {"action": "Recycle and compost household waste"},
                {"action": "Carpool to reduce emissions"},
                {"action": "Educate others about sustainability"},
            ],
        },
        {
            "template": "Maintain {object}",
            "fills": [
                {"object": "community trails"},
                {"object": "neighborhood gardens"},
                {"object": "tree wells on city blocks"},
            ],
        },
    ],
    "health": [
        {
            "template": "Donate {object}",
            "fills": [
                {"object": "blood"},
                {"object": "platelets"},
                {"object": "unused medical supplies"},
            ],
        },
        {
            "template": "Volunteer with {object}",
            "fills": [
                {"object": "community health clinics"},
                {"object": "substance recovery groups"},
            ],
        },
        {
            "template": "Support {object}",
            "fills": [
                {"object": "mental health awareness initiatives"},
            ],
        },
        {
            "template": "Organize {event}",
            "fills": [
                {"event": "a blood drive"},
                {"event": "a vaccination drive"},
                {"event": "a community fitness walk"},
                {"event": "a health screening day"},
            ],
        },
        {
            "template": "Deliver {object}",
            "fills": [
                {"object": "healthy meals to homebound neighbors"},
                {"object": "fresh produce to food pantries"},
            ],
        },
        {
            "template": "Teach {subject}",
            "fills": [
                {"subject": "first aid basics"},
                {"subject": "nutrition education"},
                {"subject": "harm-reduction safety"},
            ],
        },
    ],
    "safety": [
        {
            "template": "Report {object}",
            "fills": [
                {"object": "unsafe conditions to local authorities"},
                {"object": "broken streetlights"},
                {"object": "hazards on public trails"},
            ],
        },
        {
            "template": "Install {object}",
            "fills": [
                {"object": "smoke detectors for neighbors"},
                {"object": "reflective house numbers for emergency services"},
                {"object": "carbon monoxide alarms for seniors"},
            ],
        },
        {
            "template": "Create {object}",
            "fills": [
                {"object": "a neighborhood emergency contact list"},
                {"object": "a preparedness checklist for your block"},
            ],
        },
        {
            "template": "Participate in {object}",
            "fills": [
                {"object": "community watch programs"},
                {"object": "first responder training"},
                {"object": "CPR classes"},
            ],
        },
        {
            "template": "Advocate for {object}",
            "fills": [
                {"object": "safer crosswalks"},
                {"object": "traffic calming measures"},
                {"object": "school-zone safety"},
            ],
        },
        {
            "template": "Train {people} in {subject}",
            "fills": [
                {"people": "neighbors", "subject": "basic first aid"},
                {"people": "community volunteers", "subject": "emergency response"},
                {"people": "students", "subject": "bike safety"},
            ],
        },
    ],
    "youth": [
        {
            "template": "Mentor {people}",
            "fills": [
                {"people": "middle school students"},
                {"people": "first generation students"},
                {"people": "teens exploring careers"},
            ],
        },
        {
            "template": "Coach {activity}",
            "fills": [
                {"activity": "youth sports teams"},
                {"activity": "a youth debate club"},
                {"activity": "a youth robotics team"},
            ],
        },
        {
            "template": "Organize {event}",
            "fills": [
                {"event": "after school homework help"},
                {"event": "a youth leadership workshop"},
                {"event": "a career exploration day for students"},
            ],
        },
        {
            "template": "Provide {object}",
            "fills": [
                {"object": "backpacks with school supplies"},
                {"object": "healthy snacks for students"},
                {"object": "books for classroom libraries"},
            ],
        },
        {
            "template": "Host {event}",
            "fills": [
                {"event": "a coding club for teens"},
                {"event": "a reading hour at the library"},
                {"event": "a college prep info night for students"},
            ],
        },
    ],
    "seniors": [
        {
            "template": "Check in on {people}",
            "fills": [
                {"people": "older neighbors"},
                {"people": "seniors living alone"},
                {"people": "homebound seniors"},
            ],
        },
        {
            "template": "Deliver {object}",
            "fills": [
                {"object": "groceries to seniors"},
                {"object": "prescriptions to seniors"},
                {"object": "prepared meals to homebound neighbors"},
            ],
        },
        {
            "template": "Teach {subject} to {people}",
            "fills": [
                {"subject": "smartphone basics", "people": "seniors"},
                {"subject": "online banking safety", "people": "seniors"},
                {"subject": "video calling skills", "people": "homebound neighbors"},
            ],
        },
        {
            "template": "Help {people} {task}",
            "fills": [
                {"people": "seniors", "task": "complete benefit applications"},
                {"people": "older neighbors", "task": "schedule medical rides"},
                {"people": "homebound neighbors", "task": "tend their gardens"},
            ],
        },
        {
            "template": "Organize {event}",
            "fills": [
                {"event": "a senior social hour"},
                {"event": "a community memory cafe"},
                {"event": "a wellness check-in day"},
            ],
        },
    ],
    "arts": [
        {
            "template": "Create {object}",
            "fills": [
                {"object": "a community mural"},
                {"object": "public art for a community plaza"},
                {"object": "decorations for community events"},
            ],
        },
        {
            "template": "Organize {event}",
            "fills": [
                {"event": "a community art walk"},
                {"event": "a community open mic night"},
                {"event": "a community craft fair"},
            ],
        },
        {
            "template": "Volunteer at {object}",
            "fills": [
                {"object": "arts nonprofits"},
                {"object": "community arts centers"},
                {"object": "youth arts programs"},
            ],
        },
        {
            "template": "Support {object}",
            "fills": [
                {"object": "local arts programs"},
                {"object": "community music programs"},
                {"object": "public art initiatives"},
            ],
        },
        {
            "template": "Teach {subject} to {people}",
            "fills": [
                {"subject": "drawing basics", "people": "youth"},
                {"subject": "music fundamentals", "people": "students"},
                {"subject": "creative writing", "people": "community members"},
            ],
        },
    ],
    "transport": [
        {
            "template": "Advocate for {object}",
            "fills": [
                {"object": "safer bike lanes"},
                {"object": "accessible transit stops"},
                {"object": "well-lit sidewalks"},
            ],
        },
        {
            "template": "Report {object}",
            "fills": [
                {"object": "broken sidewalks"},
                {"object": "unsafe crosswalks"},
                {"object": "missing crosswalk signals"},
            ],
        },
        {
            "template": "Organize {event}",
            "fills": [
                {"event": "a bike safety workshop"},
                {"event": "a walking school bus program"},
                {"event": "a transit rider feedback session"},
            ],
        },
        {
            "template": "Provide {object}",
            "fills": [
                {"object": "rides to medical appointments"},
                {"object": "transit passes for job seekers"},
                {"object": "carpool help for neighbors"},
            ],
        },
        {
            "template": "Volunteer with {object}",
            "fills": [
                {"object": "a bike co-op"},
                {"object": "a safe routes to school program"},
                {"object": "community transit outreach"},
            ],
        },
    ],
    "energy": [
        {
            "template": "Weatherize {object}",
            "fills": [
                {"object": "homes for seniors"},
                {"object": "apartments for low income neighbors"},
                {"object": "community centers before winter"},
            ],
        },
        {
            "template": "Install {object}",
            "fills": [
                {"object": "energy efficient light bulbs for neighbors"},
                {"object": "smart thermostats for community spaces"},
                {"object": "weather stripping for community centers"},
            ],
        },
        {
            "template": "Educate {people} about {topic}",
            "fills": [
                {"people": "neighbors", "topic": "energy saving tips"},
                {"people": "families", "topic": "home efficiency rebates"},
                {"people": "community members", "topic": "safe heater use"},
            ],
        },
        {
            "template": "Organize {event}",
            "fills": [
                {"event": "a community energy audit day"},
                {"event": "a community insulation workshop"},
                {"event": "a community solar info session"},
            ],
        },
        {
            "template": "Reduce {object}",
            "fills": [
                {"object": "household energy use in your community"},
                {"object": "wasted energy in shared spaces"},
                {"object": "drafts in community buildings"},
            ],
        },
    ],
}


@dataclass(frozen=True)
class Candidate:
    cause: str
    text: str
    score: float
    lemmas: frozenset[str]
    signature: frozenset[str]


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def generate_candidates() -> list[tuple[str, str]]:
    candidates = []
    for cause, groups in CAUSE_TEMPLATES.items():
        for group in groups:
            template = group["template"]
            for fill in group["fills"]:
                phrase = template.format(**fill)
                candidates.append((cause, phrase))
    for phrase in SEED_BEHAVIORS:
        candidates.append(("seed", phrase))
    return candidates


def build_candidates(min_score: float = 0.3) -> list[Candidate]:
    seen = set()
    items = []
    for cause, phrase in generate_candidates():
        phrase = " ".join(phrase.split())
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        lemmas = normalize_phrase(phrase)
        score = score_from_lemmas(lemmas)
        if score < min_score:
            continue
        signature = semantic_signature(lemmas)
        items.append(
            Candidate(
                cause=cause,
                text=phrase,
                score=score,
                lemmas=frozenset(lemmas),
                signature=frozenset(signature),
            )
        )
    return items


def dedupe_candidates(
    candidates: list[Candidate],
    similarity_threshold: float,
) -> list[Candidate]:
    kept = []
    signatures = []
    for cand in sorted(candidates, key=lambda c: (-c.score, c.text)):
        if any(
            jaccard_similarity(cand.signature, sig) >= similarity_threshold
            for sig in signatures
        ):
            continue
        kept.append(cand)
        signatures.append(cand.signature)
    return kept


def mmr_select(
    candidates: list[Candidate],
    max_total: int,
    diversity: float,
    min_per_cause: int,
    max_per_cause: int,
    selected: Optional[list[Candidate]] = None,
) -> list[Candidate]:
    # Maximal marginal relevance to balance quality and diversity.
    if selected is None:
        selected = []
    remaining = list(candidates)
    cause_counts = {}
    for cand in selected:
        cause_counts[cand.cause] = cause_counts.get(cand.cause, 0) + 1

    while remaining and len(selected) < max_total:
        best = None
        best_score = -1e9
        for cand in remaining:
            if cause_counts.get(cand.cause, 0) >= max_per_cause:
                continue
            similarity = (
                max(jaccard_similarity(cand.signature, sel.signature) for sel in selected)
                if selected
                else 0.0
            )
            cause_bonus = 0.06 if cause_counts.get(cand.cause, 0) < min_per_cause else 0.0
            mmr_score = (diversity * cand.score) + cause_bonus - (
                (1.0 - diversity) * similarity
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best = cand
        if best is None:
            break
        selected.append(best)
        cause_counts[best.cause] = cause_counts.get(best.cause, 0) + 1
        remaining.remove(best)

    return selected


def select_behaviors(
    per_cause: int = 8,
    max_total: int = 80,
    min_score: float = 0.3,
) -> list[Candidate]:
    candidates = build_candidates(min_score=min_score)
    candidates = dedupe_candidates(candidates, similarity_threshold=0.8)
    by_cause = {}
    for cand in candidates:
        by_cause.setdefault(cand.cause, []).append(cand)

    num_causes = len(by_cause)
    min_per_cause = min(2, max_total // max(1, num_causes))
    seeded = []
    for cause in sorted(by_cause.keys()):
        items = dedupe_candidates(by_cause[cause], similarity_threshold=0.74)
        items.sort(key=lambda c: (-c.score, c.text))
        if min_per_cause:
            seeded.extend(items[:min_per_cause])

    seeded = dedupe_candidates(seeded, similarity_threshold=0.7)
    remaining = [cand for cand in candidates if cand not in seeded]
    selected = mmr_select(
        remaining,
        max_total=max_total,
        diversity=0.72,
        min_per_cause=min_per_cause,
        max_per_cause=max(per_cause, min_per_cause),
        selected=seeded,
    )
    selected = dedupe_candidates(selected, similarity_threshold=0.64)
    selected.sort(key=lambda c: (-c.score, c.text))
    return selected[:max_total]


def build_behavior_pool(
    min_score: float = 0.3,
    per_cause: int = 28,
    max_total: int = 500,
    min_pool: int = 160,
) -> list[Candidate]:
    pool = select_behaviors(
        per_cause=per_cause,
        max_total=max_total,
        min_score=min_score,
    )
    if len(pool) < min_pool:
        pool = select_behaviors(
            per_cause=max(20, per_cause),
            max_total=max_total,
            min_score=max(0.25, min_score - 0.05),
        )
    return pool


class BatchCycler:
    def __init__(
        self,
        pool: list[Candidate],
        batch_size: int,
        rng: random.Random,
        recent_limit: int = 200,
    ) -> None:
        self.pool = list(pool)
        self.batch_size = max(1, min(batch_size, len(self.pool)))
        self.rng = rng
        self.queue: list[Candidate] = []
        self.recent: deque[tuple[str, ...]] = deque()
        self.recent_limit = max(0, recent_limit)
        self.recent_set: set[tuple[str, ...]] = set()
        self._refill()

    def _refill(self) -> None:
        self.queue = list(self.pool)
        self.rng.shuffle(self.queue)

    def _remember(self, signature: tuple[str, ...]) -> None:
        if self.recent_limit == 0:
            return
        if len(self.recent) >= self.recent_limit:
            old = self.recent.popleft()
            self.recent_set.discard(old)
        self.recent.append(signature)
        self.recent_set.add(signature)

    def next_batch(self) -> list[Candidate]:
        if not self.pool:
            return []
        if len(self.pool) <= self.batch_size:
            return list(self.pool)

        attempts = 0
        while attempts < 8:
            if len(self.queue) < self.batch_size:
                self._refill()
            batch = self.queue[: self.batch_size]
            del self.queue[: self.batch_size]
            signature = tuple(sorted(c.text for c in batch))
            if signature not in self.recent_set:
                self._remember(signature)
                return batch
            self.queue.extend(batch)
            self.rng.shuffle(self.queue)
            attempts += 1

        self.recent.clear()
        self.recent_set.clear()
        return batch


if __name__ == "__main__":
    import time

    rng = random.Random()
    pool = build_behavior_pool()
    if not pool:
        raise SystemExit("No positive behaviors generated. Try lowering min_score.")
    batch_size = min(120, max(60, len(pool) // 3))
    if len(pool) > 1:
        batch_size = min(batch_size, len(pool) - 1)
    cycler = BatchCycler(pool, batch_size, rng, recent_limit=200)

    while True:
        for candidate in cycler.next_batch():
            print(candidate.text)
        print("\n--- Cycle complete, repeating... ---\n")
        time.sleep(2)
