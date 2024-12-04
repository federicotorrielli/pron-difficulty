from pron_difficulty import PronDifficulty

def main():
    # Initialize the evaluator
    evaluator = PronDifficulty()
    
    # Test with different languages
    words = {
        "en": ["hello", "world", "worcestershire"],
        "nb": ["kjærlighet", "øl", "språk"],
        "es": ["desarrollador", "hola", "mundo"],
        "it": ["sviluppatore", "ciao", "mondo"]
    }
    
    # Evaluate each word
    for lang, word_list in words.items():
        print(f"\n{lang.upper()} words:")
        for word in word_list:
            score = evaluator.evaluate(word, lang)
            print(f"'{word}': {score:.3f}")
            
    # Batch evaluation example
    print("\nBatch evaluation (English):")
    batch_scores = evaluator.evaluate_batch(["cat", "dog", "antidisestablishmentarianism"], "en")
    for word, score in zip(["cat", "dog", "antidisestablishmentarianism"], batch_scores):
        print(f"'{word}': {score:.3f}")

if __name__ == "__main__":
    main() 