import matplotlib
import matplotlib.pyplot as plt

from probatus.dataset import PermutationImportanceResemblance

# Turn off interactive mode in plots
plt.ioff()
matplotlib.use("Agg")


def test_permutation_resemblance_class(X1, X2, decision_tree_classifier, random_state):
    rm = PermutationImportanceResemblance(
        decision_tree_classifier, test_prc=0.5, n_jobs=1, random_state=random_state, iterations=20
    )

    actual_report, train_score, test_score = rm.fit_compute(X1, X2, return_scores=True)

    assert train_score == 1
    assert test_score == 1

    # Check report shape
    assert actual_report.shape == (3, 2)
    # Check if it is sorted by importance
    assert actual_report.iloc[0].name == "col_1"
    # Check report values
    assert actual_report.loc["col_1"]["mean_importance"] > 0
    assert actual_report.loc["col_1"]["std_importance"] > 0
    assert actual_report.loc["col_2"]["mean_importance"] == 0
    assert actual_report.loc["col_2"]["std_importance"] == 0
    assert actual_report.loc["col_3"]["mean_importance"] == 0
    assert actual_report.loc["col_3"]["std_importance"] == 0
