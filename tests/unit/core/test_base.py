from probatus.model_interpretation import ShapModelInterpreter
import pytest
from probatus._core import NotFittedError
from lightgbm import LGBMClassifier


@pytest.mark.parametrize(
    "model_class, model_params",
    [
        (LGBMClassifier, {"n_estimators": 10, "max_depth": 3, "verbose": -1}),
    ],
)
def test_fitted_exception(binary_classification_dataset, split_dataset, random_state, model_class, model_params):
    """Test that NotFittedError is raised before fitting and not after."""
    # Get training and test data
    X_train, X_test, y_train, y_test = split_dataset(binary_classification_dataset)
    class_names = ["neg", "pos"]

    # Create and fit model
    model = model_class(random_state=random_state, **model_params)
    model.fit(X_train, y_train)

    # Create SHAP interpreter
    shap_interpret = ShapModelInterpreter(model, random_state=random_state)

    # Before fit it should raise an exception
    with pytest.raises(NotFittedError):
        shap_interpret._check_if_fitted()

    # Fit the interpreter
    shap_interpret.fit(X_train, X_test, y_train, y_test, class_names=class_names)

    # Check parameters
    assert shap_interpret.fitted
    shap_interpret._check_if_fitted()  # No exception should be raised
