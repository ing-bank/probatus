import os
from sklearn.ensemble import RandomForestClassifier
from probatus.datasets import lending_club

def lending_club_model():
    """Sample Random Forest model trained on the lending club loan data.

    Model Hyper Parameberts:
        bootstrap=True, class_weight=None, criterion='gini',
        max_depth=6, max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
        oob_score=False, random_state=0, verbose=0, warm_start=False

    Returns:
        model (sklearn.ensemble.RandomForestClassifier): Pretrained model

    """
    model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        max_depth=6, max_features='auto', max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
        oob_score=False, random_state=0, verbose=0, warm_start=False)

    credit_df, x_train, x_test, y_train, y_test = lending_club()

    model = model.fit(x_train, y_train)

    return model
