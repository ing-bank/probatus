from probatus.interpret import ShapModelInterpreter
import pytest
from probatus.utils import NotFittedError


def test_fitted_exception(fitted_tree, X_train, y_train, X_test, y_test, random_state):
    """
    Test if fitted works..
    """
    class_names = ["neg", "pos"]

    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError) as _:
        shap_interpret._check_if_fitted()

    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    # Check parameters
    assert shap_interpret.fitted
    shap_interpret._check_if_fitted


@pytest.mark.xfail
def test_fitted_exception_is_raised(fitted_tree, random_state):
    """
    Test if fitted works fails when not fitted.
    """
    shap_interpret = ShapModelInterpreter(fitted_tree, random_state=random_state)

    shap_interpret._check_if_fitted
