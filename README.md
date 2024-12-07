# PronDifficulty

A rule-based system for evaluating pronunciation difficulty of words across multiple languages.

## Features

- Supports English (en), Norwegian Bokmål (nb), Spanish (es), and Italian (it)
- Provides difficulty scores from 0 (easiest) to 1 (hardest)
- Analyzes multiple linguistic aspects:
  - Phoneme complexity (40%)
  - Maximum phoneme difficulty (20%)
  - Syllable structure (20%)
  - Phonotactic constraints (10%)
  - Prosodic patterns (5%)
  - Complex phoneme ratio (5%)

## Installation

```bash
pip install pron-difficulty
```

## Quick Start

```python
from pron_difficulty import PronDifficulty

evaluator = PronDifficulty()

# Single word evaluation
score = evaluator.evaluate("difficult", "en")
print(f"Difficulty score: {score}")  # Example: 0.723

# Batch evaluation
words = ["cat", "difficult", "antidisestablishmentarianism"]
scores = evaluator.evaluate_batch(words, "en")
```

## How It Works

The system analyzes pronunciation difficulty through several components:

### 1. Phoneme Complexity (40%)

- Evaluates individual sound difficulty
- Considers articulatory features (place, manner, voicing)
- Rates rare or complex phonemes higher (e.g., /θ/, /ð/ in English)

### 2. Maximum Phoneme Difficulty (20%)

- Tracks the hardest sound in the word
- Helps identify words with even one challenging phoneme
- Example: "th" in "think" increases difficulty even if rest is simple

### 3. Syllable Structure (20%)

- Analyzes consonant clusters (e.g., "str" in "string")
- Evaluates syllable patterns (CV, CVC, CCVC, etc.)
- More complex structures = higher scores

### 4. Phonotactic Constraints (10%)

- Checks if sound combinations follow language rules
- Penalizes unusual or forbidden sequences
- Example: "ng" at start of word (uncommon in English)

### 5. Prosodic Structure (5%)

- Examines rhythm and stress patterns
- Analyzes sonority profiles
- Considers length and complexity of prosodic units

### 6. Complex Phoneme Ratio (5%)

- Percentage of difficult phonemes in word
- Helps differentiate consistently hard words
- Affects longer words more significantly

### Length Scaling

Words get additional difficulty points based on length:

- ≤3 phonemes: no extra points
- 4-6 phonemes: +0.2 per extra phoneme
- \> 6 phonemes: +0.1 per extra phoneme (max 0.7)

### Final Score Adjustment

- Simple words (score < 0.4): Gentle sigmoid curve
- Complex words (score ≥ 0.4): Steeper sigmoid + exponential boost for very complex words

## Language Support

Each supported language has:

- Custom phoneme difficulty ratings
- Language-specific syllable patterns
- Tailored phonotactic rules
- Adjusted prosodic analysis

## Examples with Explanations

```python
# English examples with typical scores:
"cat" -> 0.123         # Simple CV-C structure, common phonemes
"string" -> 0.567      # Complex onset cluster, but common in English
"rhythm" -> 0.789      # Unusual consonant patterns, no vowel between 'th' and 'm'

# Norwegian examples:
"hei" -> 0.234        # Simple structure, common sounds
"skjønnhet" -> 0.678  # Complex consonant cluster, front rounded vowel

# Spanish examples:
"casa" -> 0.123       # Simple CV-CV structure
"desarrollo" -> 0.456 # Longer but regular structure

# Italian examples:
"ciao" -> 0.234       # Simple structure despite diphthong
"struggere" -> 0.567  # Complex consonant cluster
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional language support
- Refined difficulty metrics
- Enhanced prosodic analysis
- Performance optimizations

## License

This project is licensed under the European Union Public Licence (EUPL) v. 1.2. See the [LICENSE](LICENSE) file for details.
