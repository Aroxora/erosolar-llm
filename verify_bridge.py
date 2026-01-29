
import sys
import os

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from concepts import FoundationalDictionary

def test_determinism():
    bridge = FoundationalDictionary()
    seed = 12345
    
    concepts1 = bridge.get_concepts_from_seed(seed)
    concepts2 = bridge.get_concepts_from_seed(seed)
    
    print(f"Seed {seed} -> {concepts1}")
    
    assert concepts1 == concepts2, "Bridge is not deterministic!"
    print("✅ Determinism check passed.")

def test_diversity():
    bridge = FoundationalDictionary()
    seed1 = 12345
    seed2 = 67890
    
    concepts1 = bridge.get_concepts_from_seed(seed1)
    concepts2 = bridge.get_concepts_from_seed(seed2)
    
    print(f"Seed {seed1} -> {concepts1}")
    print(f"Seed {seed2} -> {concepts2}")
    
    assert concepts1 != concepts2, "Different seeds should produce different concepts (most of the time)!"
    print("✅ Diversity check passed.")

def test_string_format():
    bridge = FoundationalDictionary()
    s = bridge.get_concept_combo_string(42, count=3)
    print(f"Formatted string: '{s}'")
    assert isinstance(s, str) and len(s) > 0
    print("✅ String formatting check passed.")

if __name__ == "__main__":
    print("Running Concept Bridge Verifications...")
    test_determinism()
    test_diversity()
    test_string_format()
