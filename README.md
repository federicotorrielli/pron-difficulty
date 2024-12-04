# Pronunciation Difficulty Evaluator

A Python package that evaluates the difficulty of pronouncing words in multiple languages using state-of-the-art techniques. Currently supports English, Norwegian, Spanish, and Italian.

## Installation

```bash
pip install pron-difficulty
```

## Usage

```python
from pron_difficulty import PronDifficulty

# Initialize the evaluator
evaluator = PronDifficulty()

# Evaluate a word's pronunciation difficulty (returns a score from 0 to 1)
difficulty = evaluator.evaluate("hello", language="en")
print(f"Difficulty score: {difficulty}")  # Example output: 0.23

# Batch evaluation
words = ["hello", "world", "difficult"]
difficulties = evaluator.evaluate_batch(words, language="en")

# Multi-language support
norwegian_score = evaluator.evaluate("kjærlighet", language="no")
spanish_score = evaluator.evaluate("desarrollador", language="es")
italian_score = evaluator.evaluate("sviluppatore", language="it")
```

## Features

- Support for multiple languages:
  - English (en)
  - Norwegian (no)
  - Spanish (es)
  - Italian (it)
- State-of-the-art phoneme-based difficulty evaluation
- Considers factors like:
  - Phoneme complexity
  - Syllable structure
  - Common pronunciation patterns
  - Language-specific features
- Fast and efficient processing
- Batch evaluation support

## How it Works

The package uses a combination of techniques to evaluate pronunciation difficulty:

1. Phoneme extraction using language-specific models
2. Syllable analysis and stress patterns
3. Statistical analysis of phoneme combinations
4. Neural network-based difficulty prediction
5. Language-specific rule evaluation
