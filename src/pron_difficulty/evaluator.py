from typing import List

import numpy as np
from phonemizer import phonemize

from .utils.phoneme_utils import (
    analyze_syllable_structure,
    get_syllable_complexity_patterns,
)
from .utils.phonological_features import (
    analyze_phonotactic_constraints,
    calculate_phonological_complexity,
    get_sonority_hierarchy_score,
)


class PronDifficulty:
    def __init__(self):
        self.phonemizer_backends = {"en": "en-us", "nb": "nb", "es": "es", "it": "it"}
        self.syllable_complexity = get_syllable_complexity_patterns()

    def _analyze_prosodic_structure(self, phonemes: List[str]) -> float:
        """
        Analyze prosodic structure using sonority hierarchy
        and optimal syllable structure
        """
        if not phonemes:
            return 0.0

        # Calculate sonority profile
        sonority_profile = [get_sonority_hierarchy_score(p) for p in phonemes]

        # Analyze sonority peaks and transitions
        peaks = 0
        valleys = 0
        steep_transitions = 0

        for i in range(1, len(sonority_profile) - 1):
            if (
                sonority_profile[i] > sonority_profile[i - 1]
                and sonority_profile[i] > sonority_profile[i + 1]
            ):
                peaks += 1
            if (
                sonority_profile[i] < sonority_profile[i - 1]
                and sonority_profile[i] < sonority_profile[i + 1]
            ):
                valleys += 1
            if abs(sonority_profile[i] - sonority_profile[i - 1]) > 0.4:
                steep_transitions += 1

        # Calculate complexity components
        syllable_count = peaks / max(3, len(phonemes) / 2)
        sonority_variance = np.std(sonority_profile)
        avg_sonority = 1 - np.mean(sonority_profile)
        transition_complexity = steep_transitions / max(3, len(phonemes) / 2)

        # Combine components with weights
        complexity = (
            0.3 * syllable_count
            + 0.3 * sonority_variance
            + 0.2 * avg_sonority
            + 0.2 * transition_complexity
        )

        # Apply exponential scaling for complex prosodic structures
        if complexity > 0.4:
            complexity = 0.4 + 0.6 * (np.exp(2 * (complexity - 0.4)) - 1)

        return min(complexity, 1.0)

    def evaluate(self, word: str, language: str) -> float:
        """
        Evaluate the pronunciation difficulty of a word using state-of-the-art
        phonological analysis.

        Args:
            word: The word to evaluate
            language: Language code ('en', 'nb', 'es', 'it')

        Returns:
            float: Difficulty score between 0 and 1
        """
        if language not in self.phonemizer_backends:
            raise ValueError(f"Unsupported language: {language}")

        # Get phonetic transcription
        phonemes = phonemize(
            word,
            backend="espeak",
            language=self.phonemizer_backends[language],
            strip=True,
        )

        # Convert phonemes string to list
        phoneme_list = list(phonemes.replace(" ", ""))

        # Calculate phonological complexity for each phoneme
        phoneme_complexities = [
            calculate_phonological_complexity(p, language) for p in phoneme_list
        ]

        # Get phoneme complexity statistics
        avg_phoneme_complexity = np.mean(phoneme_complexities)
        max_phoneme_complexity = max(phoneme_complexities)
        complex_phoneme_ratio = sum(1 for c in phoneme_complexities if c > 0.6) / len(
            phoneme_list
        )

        # Analyze syllable structure
        syllable_complexity = analyze_syllable_structure(
            phonemes, self.syllable_complexity[language]
        )

        # Analyze phonotactic constraints
        phonotactic_penalty = analyze_phonotactic_constraints(phoneme_list, language)

        # Analyze prosodic structure
        prosodic_complexity = self._analyze_prosodic_structure(phoneme_list)

        # Calculate base difficulty score with adjusted weights
        base_score = (
            0.4 * avg_phoneme_complexity  # Increased phoneme weight
            + 0.2 * max_phoneme_complexity  # Maintained difficult phonemes weight
            + 0.2 * syllable_complexity  # Maintained syllable importance
            + 0.1 * phonotactic_penalty  # Reduced phonotactic weight
            + 0.05 * prosodic_complexity  # Minimal prosodic weight
            + 0.05 * complex_phoneme_ratio  # Small boost for complex phonemes
        )

        # Different handling for simple vs complex words
        if base_score < 0.4:
            # For simple words, reduce the score and apply minimal length penalty
            length_factor = (
                0.05 * min(len(phoneme_list) - 3, 2) if len(phoneme_list) > 3 else 0
            )
            final_score = base_score * 0.5  # Reduce simple word scores
        else:
            # For complex words, apply normal length scaling
            if len(phoneme_list) <= 3:
                length_factor = 0.0
            elif len(phoneme_list) <= 6:
                length_factor = 0.2 * (len(phoneme_list) - 3)
            else:
                length_factor = 0.6 + 0.1 * (len(phoneme_list) - 6)
            length_factor = min(length_factor, 0.7)
            final_score = base_score

        # Apply length factor
        final_score *= 1 + length_factor

        # Apply sigmoid scaling with different parameters for simple vs complex words
        if final_score < 0.4:
            # Gentler curve for simple words
            x = 4 * (final_score - 0.2)
            final_score = 0.4 / (1 + np.exp(-x))
        else:
            # Steeper curve for complex words
            x = 6 * (final_score - 0.6)
            final_score = 1 / (1 + np.exp(-x))

            # Additional boost for very complex words
            if final_score > 0.7:
                final_score = 0.7 + 0.3 * (np.exp(2 * (final_score - 0.7)) - 1)

        return round(final_score, 3)

    def evaluate_batch(self, words: List[str], language: str) -> List[float]:
        """
        Evaluate pronunciation difficulty for multiple words.

        Args:
            words: List of words to evaluate
            language: Language code ('en', 'nb', 'es', 'it')

        Returns:
            List[float]: List of difficulty scores between 0 and 1
        """
        return [self.evaluate(word, language) for word in words]
