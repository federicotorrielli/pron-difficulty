from typing import Dict
from collections import defaultdict
import numpy as np


def get_default_phoneme_complexity() -> Dict[str, float]:
    """Get default phoneme complexity scores for supported languages"""
    return defaultdict(
        lambda: 0.15,
        {
            # Basic phonemes (extremely simple)
            "a": 0.01,
            "e": 0.01,
            "i": 0.01,
            "o": 0.01,
            "u": 0.01,
            "p": 0.02,
            "b": 0.02,
            "t": 0.02,
            "d": 0.02,
            "k": 0.02,
            "g": 0.02,
            "m": 0.02,
            "n": 0.02,
            "l": 0.03,
            "s": 0.03,
            # Common phonemes (still simple)
            "ɛ": 0.05,
            "ɪ": 0.05,
            "ʊ": 0.05,
            "ə": 0.03,
            "ɔ": 0.05,
            "ɑ": 0.05,
            "ŋ": 0.1,
            "r": 0.1,
            "ɾ": 0.1,
            "f": 0.05,
            "v": 0.05,
            "z": 0.05,
            "h": 0.03,
            "j": 0.05,
            "w": 0.05,
            # English specific (maintained high for difficult sounds)
            "ð": 0.9,
            "θ": 0.9,
            "ʃ": 0.7,
            "ʒ": 0.8,
            "æ": 0.6,
            "ʌ": 0.5,
            "ɜ": 0.6,
            "ɹ": 0.7,
            "tʃ": 0.7,
            "dʒ": 0.7,
            # Norwegian specific (maintained high)
            "ø": 0.8,
            "å": 0.7,
            "ç": 0.9,
            "ʉ": 0.8,
            "y": 0.7,
            "œ": 0.8,
            "ɽ": 0.9,
            "ʂ": 0.8,
            "ɖ": 0.8,
            "ɳ": 0.8,
            "ɭ": 0.8,
            # Spanish specific (maintained high)
            "ɲ": 0.7,
            "x": 0.7,
            "β": 0.6,
            "ɣ": 0.7,
            "ʝ": 0.8,
            "ʎ": 0.8,
            # Italian specific (maintained high)
            "ts": 0.8,
            "dz": 0.8,
            "kw": 0.7,
            "ɡw": 0.7,
        },
    )


def get_syllable_complexity_patterns() -> Dict[str, Dict[str, float]]:
    """Get syllable complexity patterns for supported languages"""
    return {
        "en": {
            "CCCVCC": 1.0,  # 'strengths'
            "CCCVC": 0.9,  # 'splits'
            "CCVCC": 0.8,  # 'tanks'
            "CCVC": 0.6,  # 'stop'
            "CVC": 0.08,  # 'cat' - drastically reduced
            "CV": 0.03,  # 'to'
            "VC": 0.05,  # 'at'
            "V": 0.01,  # 'a'
        },
        "nb": {
            "CCVCC": 1.0,  # 'språk'
            "CCVC": 0.8,  # 'stor'
            "CCV": 0.7,  # 'kje'
            "CVC": 0.1,  # 'tak'
            "CV": 0.05,  # 'ja'
            "VC": 0.08,  # 'øl'
        },
        "es": {
            "CCVCC": 0.9,  # 'trans'
            "CCVC": 0.7,  # 'tres'
            "CVC": 0.08,  # 'pan'
            "CV": 0.03,  # 'la'
            "V": 0.01,  # 'a'
        },
        "it": {
            "CCVCC": 0.9,  # 'sport'
            "CCVC": 0.7,  # 'stop'
            "CVC": 0.1,  # 'per'
            "CV": 0.05,  # 'si'
            "V": 0.01,  # 'e'
        },
    }


def analyze_syllable_structure(phonemes: str, patterns: Dict[str, float]) -> float:
    """
    Analyze syllable structure complexity with improved pattern matching
    """
    # Convert phonemes to CV pattern
    cv_pattern = ""
    vowels = set("aeiouæøåɑɛɪʊəɜœʉyɔ")
    for p in phonemes:
        if p in vowels:
            cv_pattern += "V"
        else:
            cv_pattern += "C"

    # Find all matching patterns and their complexities
    pattern_scores = []
    for pattern, score in patterns.items():
        count = cv_pattern.count(pattern)
        if count > 0:
            # Increase score for multiple occurrences of complex patterns
            if score > 0.5:  # Only boost complex patterns
                pattern_scores.append(score * (1 + 0.2 * (count - 1)))
            else:
                # For simple patterns, reduce score with multiple occurrences
                pattern_scores.append(score * (1 / (1 + 0.1 * (count - 1))))

    # Calculate complexity based on the most complex patterns found
    if pattern_scores:
        # Weight the highest pattern more heavily for complex patterns
        max_score = max(pattern_scores)
        if max_score > 0.5:
            complexity = (
                max_score * 0.9 + (sum(pattern_scores) / len(pattern_scores)) * 0.1
            )
        else:
            # For simple patterns, use average to avoid overweighting
            complexity = (max_score + sum(pattern_scores) / len(pattern_scores)) * 0.5
    else:
        complexity = 0.01  # Extremely low base complexity

    # Add progressive length penalty
    length = len(phonemes)
    if length <= 3:
        length_penalty = 0.0
    elif length <= 6:
        length_penalty = 0.15 * (length - 3)
    else:
        length_penalty = 0.45 + 0.1 * (length - 6)

    # Cap the length penalty
    length_penalty = min(length_penalty, 0.6)

    # Apply exponential scaling for complex patterns
    if complexity > 0.5:
        complexity = 0.5 + 0.5 * (np.exp(2 * (complexity - 0.5)) - 1)

    # Different weighting for simple vs complex patterns
    if complexity < 0.3:
        final_score = (
            complexity * 0.9 + length_penalty * 0.1
        )  # Less length influence for simple words
    else:
        final_score = (
            complexity * 0.7 + length_penalty * 0.3
        )  # More length influence for complex words

    # Final exponential scaling for very difficult combinations
    if final_score > 0.6:
        final_score = 0.6 + 0.4 * (np.exp(2 * (final_score - 0.6)) - 1)

    return min(final_score, 1.0)
