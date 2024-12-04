from typing import List
import numpy as np

# Based on Hayes' Feature Geometry and modern phonological theory
FEATURE_VECTORS = {
    # Place features
    "LABIAL": ["p", "b", "m", "f", "v", "w", "ʍ", "β"],
    "CORONAL": ["t", "d", "n", "s", "z", "ʃ", "ʒ", "θ", "ð", "ɾ", "r", "l", "ɹ"],
    "DORSAL": ["k", "g", "ŋ", "x", "ɣ", "ʎ", "j"],
    # Manner features
    "STOP": ["p", "b", "t", "d", "k", "g", "ʔ"],
    "FRICATIVE": ["f", "v", "s", "z", "ʃ", "ʒ", "θ", "ð", "x", "ɣ", "h", "ç", "ʂ"],
    "AFFRICATE": ["tʃ", "dʒ", "ts", "dz"],
    "NASAL": ["m", "n", "ŋ", "ɲ", "ɳ"],
    "LIQUID": ["l", "r", "ɾ", "ɹ", "ʎ", "ɭ"],
    "GLIDE": ["j", "w"],
    # Vowel features
    "HIGH": ["i", "y", "ɨ", "ʉ", "ɯ", "u", "ɪ", "ʊ"],
    "MID": ["e", "ø", "ɘ", "ɵ", "ɤ", "o", "ə", "ɛ", "œ", "ʌ", "ɔ"],
    "LOW": ["æ", "a", "ɶ", "ɑ", "ɒ"],
    "FRONT": ["i", "y", "e", "ø", "ɛ", "œ", "æ", "ɪ"],
    "CENTRAL": ["ɨ", "ʉ", "ɘ", "ɵ", "ə", "ɜ", "ɐ"],
    "BACK": ["ɯ", "u", "ɤ", "o", "ʌ", "ɔ", "ɑ", "ɒ", "ʊ"],
    "ROUNDED": ["y", "ʉ", "u", "ø", "ɵ", "o", "œ", "ɔ", "ɒ", "ʊ"],
}

# Language-specific marked features with increased difficulty scores
MARKED_FEATURES = {
    "en": {
        "θ": 0.95,  # dental fricatives (extremely hard for most L2 learners)
        "ð": 0.95,
        "ɹ": 0.85,  # approximant r (unique to English)
        "w": 0.75,  # labial-velar approximant
        "æ": 0.85,  # near-low front unrounded (very difficult)
        "ə": 0.70,  # schwa (in unstressed positions)
    },
    "nb": {
        "ç": 0.90,  # voiceless palatal fricative (very difficult)
        "ʉ": 0.90,  # close central rounded (rare cross-linguistically)
        "y": 0.85,  # close front rounded
        "ø": 0.85,  # mid front rounded
        "ɽ": 0.95,  # retroflex flap (extremely marked)
    },
    "es": {
        "r": 0.90,  # trilled r (difficult for most L2 learners)
        "x": 0.80,  # voiceless velar fricative
        "β": 0.85,  # bilabial approximant
        "ð": 0.85,  # dental approximant
        "ɣ": 0.85,  # velar approximant
    },
    "it": {
        "ʎ": 0.85,  # palatal lateral
        "ɲ": 0.80,  # palatal nasal
        "r": 0.85,  # trilled r
        "kw": 0.80,  # labialized velar
        "ts": 0.85,  # alveolar affricates
        "dz": 0.85,
    },
}


def get_feature_vector(phoneme: str) -> np.ndarray:
    """Convert a phoneme to its feature vector representation"""
    vector = np.zeros(len(FEATURE_VECTORS))
    for i, (feature, phonemes) in enumerate(FEATURE_VECTORS.items()):
        if phoneme in phonemes:
            vector[i] = 1
    return vector


def get_sonority_hierarchy_score(phoneme: str) -> float:
    """
    Calculate sonority score based on the universal sonority hierarchy
    Vowels > Glides > Liquids > Nasals > Fricatives > Stops
    """
    if phoneme in FEATURE_VECTORS["STOP"]:
        return 0.1
    elif phoneme in FEATURE_VECTORS["FRICATIVE"]:
        return 0.3
    elif phoneme in FEATURE_VECTORS["AFFRICATE"]:
        return 0.2
    elif phoneme in FEATURE_VECTORS["NASAL"]:
        return 0.5
    elif phoneme in FEATURE_VECTORS["LIQUID"]:
        return 0.7
    elif phoneme in FEATURE_VECTORS["GLIDE"]:
        return 0.8
    elif any(phoneme in v for v in ["HIGH", "MID", "LOW"]):  # Vowels
        return 1.0
    return 0.4  # Default value


def calculate_phonological_complexity(phoneme: str, language: str) -> float:
    """
    Calculate phonological complexity with increased penalties for marked features
    """
    # Get feature vector
    features = get_feature_vector(phoneme)
    feature_complexity = np.sum(features) / len(features)

    # Check if phoneme is marked in the target language
    markedness = MARKED_FEATURES[language].get(phoneme, 1.0)

    # Get sonority score
    sonority = get_sonority_hierarchy_score(phoneme)

    # Calculate base complexity
    complexity = (
        0.4 * feature_complexity
        + 0.4 * (1 - sonority)  # Inverse of sonority
        + 0.2 * markedness
    )

    # Apply exponential scaling for very complex phonemes
    if complexity > 0.7:
        complexity = 0.7 + 0.3 * (np.exp(complexity - 0.7) - 1)

    return min(complexity, 1.0)


def analyze_phonotactic_constraints(phonemes: List[str], language: str) -> float:
    """
    Analyze violation of phonotactic constraints with increased penalties
    """
    violations = 0
    total_constraints = 0

    # Universal constraints
    if len(phonemes) >= 2:
        for i in range(len(phonemes) - 1):
            # Sonority Sequencing Principle
            curr_sonority = get_sonority_hierarchy_score(phonemes[i])
            next_sonority = get_sonority_hierarchy_score(phonemes[i + 1])

            # Increased penalty for sonority violations
            if curr_sonority <= next_sonority:
                if all(
                    p in FEATURE_VECTORS["STOP"] + FEATURE_VECTORS["FRICATIVE"]
                    for p in [phonemes[i], phonemes[i + 1]]
                ):
                    violations += 1.5  # Increased from 1.0

            # Penalize complex consonant clusters more heavily
            if i < len(phonemes) - 2:
                if all(p not in ["a", "e", "i", "o", "u"] for p in phonemes[i : i + 3]):
                    violations += 1.0

            total_constraints += 1

    # Language-specific constraints with higher penalties
    if language == "es":
        # Spanish doesn't allow word-initial sC clusters
        if (
            len(phonemes) >= 2
            and phonemes[0] == "s"
            and phonemes[1] in FEATURE_VECTORS["STOP"]
        ):
            violations += 2.0  # Increased from 1.0
    elif language == "it":
        # Italian geminates
        if (
            len(phonemes) >= 2
            and phonemes[-1] in FEATURE_VECTORS["STOP"]
            and phonemes[-2] in FEATURE_VECTORS["STOP"]
        ):
            violations += 1.5  # Increased from 1.0

    total_constraints = max(total_constraints, 1)

    # Apply exponential scaling for multiple violations
    penalty = violations / total_constraints
    if penalty > 0.5:
        penalty = 0.5 + 0.5 * (np.exp(penalty - 0.5) - 1)

    return min(penalty, 1.0)
