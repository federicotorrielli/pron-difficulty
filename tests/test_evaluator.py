import pytest
from pron_difficulty import PronDifficulty

@pytest.fixture
def evaluator():
    return PronDifficulty()

def test_basic_evaluation(evaluator):
    # Test English
    score_en = evaluator.evaluate("hello", "en")
    assert 0 <= score_en <= 1
    
    # Test Norwegian
    score_no = evaluator.evaluate("kjærlighet", "no")
    assert 0 <= score_no <= 1
    
    # Test Spanish
    score_es = evaluator.evaluate("desarrollador", "es")
    assert 0 <= score_es <= 1
    
    # Test Italian
    score_it = evaluator.evaluate("sviluppatore", "it")
    assert 0 <= score_it <= 1

def test_batch_evaluation(evaluator):
    words = ["hello", "world", "test"]
    scores = evaluator.evaluate_batch(words, "en")
    assert len(scores) == len(words)
    assert all(0 <= score <= 1 for score in scores)

def test_invalid_language(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate("hello", "invalid_lang")

def test_relative_difficulty(evaluator):
    # Test that known difficult words score higher than simple ones
    simple_word = evaluator.evaluate("cat", "en")
    difficult_word = evaluator.evaluate("worcestershire", "en")
    assert simple_word < difficult_word 