import numpy as np
from probatus.utils.tree import TreePathFinder
from probatus.utils.warnings import NotIntendedUseWarning
import pytest


class mock_sub_tree():
    def __init__(self, node_count, cl, cr, feat, thresh):
        self.node_count = node_count
        self.children_left = cl
        self.children_right = cr
        self.feature = feat
        self.threshold = thresh

class mock_tree():
    def __init__(self, node_count, cl, cr, feat, thresh):
        self.tree_ = mock_sub_tree(node_count, cl, cr, feat, thresh)

@pytest.fixture(scope='function')
def make_tree_mock():
    return mock_tree(
        node_count=7,
        cl=np.array([1, -1, 3, 4, -1, -1, -1]),
        cr=np.array([2, -1, 6, 5, -1, -1, -1]),
        feat=np.array([0, -2, 0, 0, -2, -2, -2]),
        thresh=np.array([-3.54, -2., 2.795, -1.985, -2., -2., -2.])
    )

@pytest.fixture(scope='function')
def make_complex_tree_mock():
    return mock_tree(
        node_count=15,
        cl=np.array([1, -1, 3, 4, 5, -1, -1, 8, -1, -1, 11, 12, -1, -1, -1]),
        cr=np.array([2, -1, 10, 7, 6, -1, -1, 9, -1, -1, 14, 13, -1, -1, -1]),
        feat=np.array([0, -2, 0, 0, 0, -2, -2, 0, -2, -2, 0, 0, -2, -2, -2]),
        thresh=np.array([-3.54, -2, 2.795, -1.985, -2.785, -2., -2., 1.675, -2., -2., 3.925, 3.865, -2., -2., -2.])
    )

@pytest.fixture(scope='function')
def make_wrong_tree_mock():
    return mock_tree(
        node_count=7,
        cl=np.array([1, -1, 3, 4, -1, -1, -1]),
        cr=np.array([2, -1, 6, 5, -1, -1, -1]),
        feat=np.array([0, -2, 0, 1, -2, -2, -2]), # this tree is trained on two features, not ideal for binning
        thresh=np.array([-3.54, -2., 2.795, -1.985, -2., -2., -2.])
    )


def test_simple_tree(make_tree_mock):
    # Quick interpretation tip:
    # The sklearn trees structure is represented by the arrays described in the mock tree.
    # When the index is negative, this means that the leaf is a terminal leaf, otherwise is part of the node
    # children left/right contain the index of the next leaf, while feat contains the index of the feature used.
    # In this toy case, the leaf are at index 1,4,5,6 (corresponding to the negative indexes in the arrays).
    # Furthermore, the order of the expected boundaries can be extracted from the thesholds

    # test_tree_1 =

    tpf1 = TreePathFinder(make_tree_mock)


    exp_boundaries = {1: {'min': -np.inf, 'max': -3.54},
                      4: {'min': -3.54, 'max': -1.985},
                      5: {'min': -1.985, 'max': 2.795},
                      6: {'min': 2.795, 'max': np.inf}}


    assert tpf1.get_boundaries() == exp_boundaries
    assert all(tpf1.is_leaves[[1,4,5,6]])#

def test_complex_tree(make_complex_tree_mock):


    tpf2 = TreePathFinder(make_complex_tree_mock)

    exp_boundaries = {1: {'min': -np.inf, 'max': -3.54},
                      5: {'min': -3.54, 'max': -2.785},
                      6: {'min': -2.785, 'max': -1.985},
                      8: {'min': -1.985, 'max': 1.675},
                      9: {'min': 1.675, 'max': 2.795},
                      12: {'min': 2.795, 'max': 3.865},
                      13: {'min': 3.865, 'max': 3.925},
                      14: {'min': 3.925, 'max': np.inf}}

    assert tpf2.get_boundaries() == exp_boundaries
    assert all(tpf2.is_leaves[[1, 5, 6, 8,9,12,13,14]])

def test_warning_raised(make_wrong_tree_mock):
    # This specific tree is trained on two features (feat array has index 0 and 1)
    # The boundaries found by the TreePathFinder are indended for use only on 1 dimensions
    # Hence this tests that a warning is being raised in case this happens

    with pytest.warns(NotIntendedUseWarning):
        TreePathFinder(make_wrong_tree_mock).get_boundaries()

