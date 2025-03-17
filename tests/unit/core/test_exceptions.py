import pytest
from probatus.core import NotFittedError


def test_not_fitted_error():
    """Test NotFittedError initialization, behavior, and inheritance."""
    error_message = "The estimator has not been fitted"
    with pytest.raises(NotFittedError) as excinfo:
        raise NotFittedError(error_message)

    assert excinfo.value.message == error_message
    assert str(excinfo.value) == error_message
