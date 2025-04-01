# TODO: Make this a base class which contains all kinds of extensive validations for:
# - model
# - cv
# - pipeline
# - parameter validations
# - early stopping

# From here all the other classes interact with outside.
# This'll be a Estimator/model object that is thus a wrapper around all things estimator related.


from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


class EstimatorObject:
    # TODO: Param should be model, cv, etc
    def __init__(
        self,
        model: BaseEstimator,
        cv: BaseCrossValidator,
        parameter_grid: dict,
        early_stopping: bool,
    ):
        self.model = model
        self.cv = cv
        self.parameter_grid = parameter_grid
        self.early_stopping = early_stopping
