import pytest
from sklearn.metrics import make_scorer, accuracy_score

from probatus.utils.scoring import get_single_scorer, Scorer


def test_get_single_scorer():
    """Test get_single_scorer function with different input types."""
    # Test with string metric name
    scorer = get_single_scorer("accuracy")
    assert isinstance(scorer, Scorer)
    assert scorer.metric_name == "accuracy"

    # Test with Scorer object
    original_scorer = Scorer("accuracy")
    returned_scorer = get_single_scorer(original_scorer)
    assert original_scorer is returned_scorer
    assert isinstance(returned_scorer, Scorer)


def test_invalid_scorer_input():
    """Test ValueError is raised for invalid input."""
    # Test with invalid input
    with pytest.raises(ValueError):
        get_single_scorer(123)

    # Test with invalid metric name
    with pytest.raises(ValueError):
        Scorer("invalid_metric_name")


def test_scorer_score_methods(sample_data):
    """Test the score method of Scorer with both standard and custom metrics."""
    X_df, y_series, model = sample_data

    # Test with a standard metric
    scorer = Scorer("accuracy")
    assert isinstance(scorer, Scorer)
    assert scorer.metric_name == "accuracy"
    score = scorer.score(model, X_df, y_series)

    # The score should be a float between 0 and 1 for accuracy
    assert isinstance(score, float)
    assert 0 <= score <= 1

    # Verify the score matches what we'd expect
    expected_score = accuracy_score(y_series, model.predict(X_df))
    assert score == pytest.approx(expected_score)

    # Test with a custom metric
    def custom_count(y_true, y_pred):
        return (y_true == y_pred).sum()

    custom_scorer = make_scorer(custom_count)
    scorer = Scorer("custom_count", custom_scorer=custom_scorer)
    assert scorer.metric_name == "custom_count"
    assert scorer.scorer is custom_scorer
    score = scorer.score(model, X_df, y_series)

    # Verify the score matches what we'd expect
    expected_score = (y_series == model.predict(X_df)).sum()
    assert score == pytest.approx(expected_score)
