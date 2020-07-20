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



def test_simple_tree():
    test_tree_1 = mock_tree(
        node_count=7,
        cl=np.array([1, -1, 3, 4, -1, -1, -1]),
        cr=np.array([2, -1, 6, 5, -1, -1, -1]),
        feat=np.array([0, -2, 0, 0, -2, -2, -2]),
        thresh=np.array([-3.54, -2., 2.795, -1.985, -2., -2., -2.])
    )

    tpf1 = TreePathFinder(test_tree_1)

    exp_boundaries = {1: {'min': -np.inf, 'max': -3.54},
                      4: {'min': -3.54, 'max': -1.985},
                      5: {'min': -1.985, 'max': 2.795},
                      6: {'min': 2.795, 'max': np.inf}}

    assert tpf1.get_boundaries() == exp_boundaries
    assert all(tpf1.is_leaves[[1,4,5,6]])#

def test_warning_raised():

    test_tree_2 = mock_tree(
        node_count = 7,
        cl = np.array([ 1, -1,  3,  4, -1, -1, -1]),
        cr = np.array([ 2, -1,  6,  5, -1, -1, -1]),
        feat = np.array([ 0, -2,  0,  1, -2, -2, -2]),
        thresh = np.array([-3.54, -2.,  2.795, -1.985, -2., -2., -2.])
    )

    with pytest.warns(NotIntendedUseWarning):
        TreePathFinder(test_tree_2).get_boundaries()

