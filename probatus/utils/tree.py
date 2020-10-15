# Copyright (c) 2020 ING Bank N.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from collections import defaultdict
import numpy as np
from tqdm import tqdm
import warnings
from .warnings import  NotIntendedUseWarning


class TreePathFinder():
    """Class to calculate the boundaries of a decision tree.

    It retrieves the structure from the decision tree

    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.n_nodes = estimator.tree_.node_count
        self.children_left = estimator.tree_.children_left
        self.children_right = estimator.tree_.children_right
        self.feature = estimator.tree_.feature
        self.threshold = estimator.tree_.threshold
        self.is_leaves = self._find_leaves()
        self.decision_path = self.find_decision_to_leaves()
        self.bin_boundaries = self.find_bin_boundaries()


    def _find_leaves(self):
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        n_nodes = self.n_nodes
        children_left = self.children_left
        children_right = self.children_right

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        return is_leaves

    def find_parent(self, leaf_id):

        in_left = np.where(self.children_left == leaf_id)[0]
        in_right = np.where(self.children_right == leaf_id)[0]
        is_inleft = len(in_left) > 0
        is_inright = len(in_right) > 0

        if is_inleft & is_inright:
            raise ValueError(f"leaf with id {leaf_id} not found in tree")
        elif is_inleft:
            parent = in_left[0]
            operator = '<='
        elif is_inright:
            parent = in_right[0]
            operator = '>'
        else:
            parent = 0
            operator = 'None'

        threshold = self.threshold[parent]
        feature = self.feature[parent]

        return parent, threshold, operator, feature


    def find_decision_to_leaves(self):

        leaves_ids = np.where(self.is_leaves)[0]

        decision_path = defaultdict(list)
        for leaf_id in tqdm(leaves_ids):
            parent_id = -1
            node_id = leaf_id
            while parent_id != 0:
                path_step = self.find_parent(node_id)
                parent_id = path_step[0]
                decision_path[leaf_id].append(path_step)
                node_id = parent_id

        return decision_path

    def find_bin_boundaries(self):

        out_dict = dict()
        for leaf_id in self.decision_path.keys():
            one_leaf_decisions = self.decision_path[leaf_id]

            if '<=' not in [oper[2] for oper in one_leaf_decisions]:
                max_val = np.inf
            else:
                max_val = min([oper[1] for oper in one_leaf_decisions if oper[2] == '<='])

            if '>' not in [oper[2] for oper in one_leaf_decisions]:
                min_val = -np.inf
            else:
                min_val = max([oper[1] for oper in one_leaf_decisions if oper[2] == '>'])


            out_dict[leaf_id] = {
                'min': min_val,
                'max': max_val,
            }

        return out_dict


    def get_boundaries(self):

        # check how many features are there. There is always a unique negative value in the array of features
        # that corresponds to the index of the leaves.
        # Hence the total number of features in the tree is the length of the array -1
        n_features = len(np.unique(self.feature))-1

        if n_features>1:
            warning = f"This functionality is intended for trees fitted on 1 feature. The current tree is fitted " \
                f"with {n_features} features"
            warnings.warn(NotIntendedUseWarning(warning))

        return self.bin_boundaries