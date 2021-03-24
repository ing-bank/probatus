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


from ..utils import NotFittedError, UnsupportedModelError, BaseFitComputeClass
import numpy as np
import pandas as pd
import copy

from sklearn.cluster import KMeans
from probatus.utils import shap_helpers


def return_confusion_metric(y_true, y_score, normalize=False):
    """
    Computes a confusion metric as absolute difference between the y_true and y_score.

    If normalize is set to true, it will normalize y_score to the maximum value in the array

    Args:
        y_true: (np.ndarray or pd.Series) true targets
        y_score: (np.ndarray or pd.Series) model output
        normalize: boolean, normalize or not to the maximum value

    Returns: (np.ndarray or pd.Series) confusion metric

    """

    if normalize:
        y_score = y_score / y_score.max()

    return np.abs(y_true - y_score)


class BaseInspector(BaseFitComputeClass):
    """
    Base class.
    """

    def __init__(self, algotype, **kwargs):
        """
        Init.
        """
        self.algotype = algotype
        # TODO fix compilatiopn issue on  for hdbscan
        # if algotype =='dbscan':
        #     self.clusterer = hdbscan.HDBSCAN(prediction_data=True,**kwargs)
        if algotype == "kmeans":
            self.clusterer = KMeans(**kwargs)
        else:
            raise UnsupportedModelError("The algorithm {} is not supported".format(algotype))

    def __repr__(self):
        """
        String representation.
        """
        repr_ = "{},\n\t{}".format(self.__class__.__name__, self.algotype)
        if self.fitted:
            repr_ += "\n\tTotal clusters {}".format(np.unique(self.clusterer.labels_).shape[0])
        return repr_

    def fit_clusters(self, X):
        """
        Perform the fit of the clusters with the algorithm specified in the constructor.

        Args:
            X: input features

        Returns: cluster labels
        """
        self.clusterer.fit(X)
        self.fitted = True

        return self

    def predict_clusters(self, X):
        """
        Predict clusters.
        """
        if not self.fitted:
            raise NotFittedError("Inspector not fitter. Run .fit()")

        labels = None
        if self.algotype == "kmeans":
            labels = self.clusterer.predict(X)
        if self.algotype == "dbscan":
            raise NotImplementedError("Implementation not finished (note the hdbscan package is not imported yet!)")
            # labels, strengths = hdbscan.approximate_predict(self.clusterer, X)

        return labels

    @staticmethod
    def assert_is_dataframe(df):
        """
        Assertion.
        """
        if isinstance(df, pd.DataFrame):
            return df

        elif isinstance(df, np.ndarray) and len(df.shape) == 2:
            return pd.DataFrame(df)

        else:
            raise NotImplementedError("Sorry, X needs to be a pd.DataFrame for for a 2 dimensional numpy array")

    @staticmethod
    def assert_is_series(series, index=None):
        """
        Assert input is a pandas series.
        """
        if isinstance(series, pd.Series):
            return series
        elif isinstance(series, pd.DataFrame) and series.shape[1] == 1:
            return pd.Series(series.values.ravel(), index=series.index)
        elif isinstance(series, np.ndarray) and len(series.shape) == 1 and index is not None:
            return pd.Series(series, index=index)
        else:
            raise TypeError(
                "The object should be a pd.Series, a dataframe with one collumn or a 1 dimensional numpy array"
            )


class InspectorShap(BaseInspector):
    """
    Class to perform inspection of the model prediction based on Shapley values.

    It uses the calculated Shapley values for the train model to build clusters in the shap space.
    For each cluster, an average confusion, average predicted probability and observed rate of a single class is
    calculated.
    Every sub cluster can be retrieved with the function slice_cluster to perform deeper analysis.

    The original dataframe indexing is used in slicing the dataframe, ensuring easy filtering

    Args:
            model: (obj) pretrained model (with sklearn-like API)
            algotype: (str) clustering algorithm (supported are kmeans and hdbscan)
            confusion_metric: (str) Confusion metric to use:
                - "proba": it will calculate the confusion metric as the absolute value of the target minus
                    the predicted probability. This provides a continuous measure of confusion, where 0 indicated
                    correct predictions and the closer the number is to 1, the higher the confusion
            normalize_probability: (boolean) if true, it will normalize the probabilities to the max value when computing
                the confusion metric
            cluster_probabilities: (boolean) if true, uses the model prediction as an input for the cluster prediction
            **kwargs: keyword arguments for the clustering algorithm

    """  # noqa

    def __init__(
        self,
        model,
        algotype="kmeans",
        confusion_metric="proba",
        normalize_probability=False,
        cluster_probability=False,
        **kwargs
    ):
        """
        Init.
        """
        super().__init__(algotype, **kwargs)
        self.model = model
        self.isinspected = False
        self.hasmultiple_dfs = False
        self.normalize_proba = normalize_probability
        self.cluster_probabilities = cluster_probability
        self.agg_summary_df = None
        self.set_names = None
        self.confusion_metric = confusion_metric
        self.cluster_report = None
        self.y = None
        self.predicted_proba = None
        self.X_shap = None
        self.clusters = None
        self.init_eval_set_report_variables()

        if confusion_metric not in ["proba"]:
            # TODO implement the target method
            raise NotImplementedError("confusion metric {} not supported. See docstrings".format(confusion_metric))

    def __repr__(self):
        """
        String representation.
        """
        repr_ = "{},\n\t{}".format(self.__class__.__name__, self.algotype)
        if self.fitted:
            repr_ += "\n\tTotal clusters {}".format(np.unique(self.clusterer.labels_).shape[0])
        return repr_

    def init_eval_set_report_variables(self):
        """
        Init report values.
        """
        self.X_shaps = list()
        self.clusters_list = list()
        self.ys = list()
        self.predicted_probas = list()

    def compute_probabilities(self, X):
        """
        Compute the probabilities for the model using the sklearn API.

        Args:
            X: Feature set

        Returns: (np.array) probability
        """
        return self.model.predict_proba(X)[:, 1]

    def fit_clusters(self, X):
        """
        Perform the fit of the clusters with the algorithm specified in the constructor.

        Args:
            X: input features
        """
        X = copy.deepcopy(X)

        if self.cluster_probabilities:
            X["probs"] = self.predicted_proba

        return super().fit_clusters(X)

    def predict_clusters(self, X):
        """
        Predicts the clusters of the dataset X.

        Args:
            X: features

        Returns: cluster labels
        """
        X = copy.deepcopy(X)

        if self.cluster_probabilities:
            X["probs"] = self.predicted_proba

        return super().predict_clusters(X)

    def fit(self, X, y=None, eval_set=None, sample_names=None, **shap_kwargs):
        """
        Fits and orchestrates the cluster calculations.

        Args:
            X: (pd.DataFrame) with the features set used to train the model
            y: (pd.Series, default=None): targets used to train the model
            eval_set: (list, default=None). list of tuples in the shape (X,y) containing evaluation samples, for example
                a test sample, validation sample etc... X corresponds to the feature set of the sample, y corresponds
                to the targets of the samples
            sample_names: (list of strings, default=None): list of suffixed for the samples.
                If none, it will be labelled with
                sample_{i}, where i corresponds to the index of the sample.
                List length must match that of eval_set
            **shap_kwargs:  kwargs to pass to the Shapley Tree Explained
        """
        self.set_names = sample_names
        if sample_names is not None:
            # Make sure that the amount of eval sets matches the set names
            assert len(eval_set) == len(sample_names), "set_names must be the same length as eval_set"

        (
            self.y,
            self.predicted_proba,
            self.X_shap,
            self.clusters,
        ) = self.perform_fit_calc(X=X, y=y, fit_clusters=True, **shap_kwargs)
        if eval_set is not None:
            assert isinstance(eval_set, list), "eval_set needs to be a list"

            self.hasmultiple_dfs = True
            # Reset lists in case inspect run multiple times
            self.init_eval_set_report_variables()

            for X_, y_ in eval_set:
                y_, predicted_proba_, X_shap_, clusters_ = self.perform_fit_calc(
                    X=X_, y=y_, fit_clusters=False, **shap_kwargs
                )

                self.X_shaps.append(X_shap_)
                self.ys.append(y_)
                self.predicted_probas.append(predicted_proba_)
                self.clusters_list.append(clusters_)

        return self

    def perform_fit_calc(self, X, y, fit_clusters=False, **shap_kwargs):
        """
        Performs cluster calculations for a specific X and y.

        Args:
            X: pd.DataFrame with the features set used to train the model
            y: pd.Series (default None): targets used to train the model
            fit_clusters: flag indicating whether clustering algorithm should be trained with computed shap values
            **shap_kwargs:  kwargs to pass to the Shapley Tree Explained
        """
        X = self.assert_is_dataframe(X)
        y = self.assert_is_series(y, index=X.index)

        # Compute probabilities for the input X using model
        predicted_proba = pd.Series(self.compute_probabilities(X), index=y.index, name="pred_proba")

        # Compute SHAP values and cluster them
        X_shap = shap_helpers.shap_to_df(self.model, X, **shap_kwargs)
        if fit_clusters:
            self.fit_clusters(X_shap)
        clusters = pd.Series(self.predict_clusters(X_shap), index=y.index, name="cluster_id")
        return y, predicted_proba, X_shap, clusters

    def _compute_report(self):
        """
        Helper function to compute the report of the inspector.

        Performs aggregations per cluster id
        """
        self.summary_df = self.create_summary_df(
            self.clusters, self.y, self.predicted_proba, normalize=self.normalize_proba
        )
        self.agg_summary_df = self.aggregate_summary_df(self.summary_df)

        if self.hasmultiple_dfs:

            self.summary_dfs = [
                self.create_summary_df(clust, y, pred_proba, normalize=self.normalize_proba)
                for clust, y, pred_proba in zip(self.clusters_list, self.ys, self.predicted_probas)
            ]

            self.agg_summary_dfs = [self.aggregate_summary_df(df) for df in self.summary_dfs]

    def compute(self):
        """
        Calculates a report containing the information per cluster.

        Includes the following:
            - cluster id
            - total number of observations in the cluster
            - total number of target 1 in the cluster
            - target 1 rate (ration of target 1 counts/observations)
            - average predicted probabilitites
            - average confusion

        If multiple eval_sets were passed in the inspect() functions, the output will contain those aggregations as well.
        The output names will use the sample names provided in the inspect function. Otherwise they will be labelled by
        the suffix sample_{i}, where i is the index of the sample

        Returns: (pd.DataFrame) with above mentioned aggregations.
        """
        if self.cluster_report is not None:
            return self.cluster_report

        self._compute_report()
        out = copy.deepcopy(self.agg_summary_df)

        if self.hasmultiple_dfs:

            for ix, agg_summary_df in enumerate(self.agg_summary_dfs):
                if self.set_names is None:
                    sample_suffix = "sample_{}".format(ix + 1)
                else:
                    sample_suffix = self.set_names[ix]

                out = pd.merge(
                    out,
                    agg_summary_df,
                    how="left",
                    on="cluster_id",
                    suffixes=("", "_{}".format(sample_suffix)),
                )

        self.cluster_report = out
        return self.cluster_report

    def slice_cluster(
        self,
        cluster_id,
        summary_df=None,
        X_shap=None,
        y=None,
        predicted_proba=None,
        complementary=False,
    ):
        """
        Slices the input dataframes by the cluster.

        Args:
            cluster_id: (int or list for multiple cluster_id) cluster ids to to slice
            summary_df: Optional parameter - the summary_df on which the masking should be performed.
                if not passed the slicing is performed on summary generated by inspect method on X and y
            X_shap: Optional parameter - the SHAP values generated from on X on which the masking should be performed.
                if not passed the slicing is performed on X_shap generated by inspect method on X and y
            y: Optional parameter - the y on which the masking should be performed.
                if not passed the slicing is performed on y passed to inspect
            predicted_proba: Optional parameter - the predicted_proba on which the masking should be performed.
                if not passed the slicing is performed on predicted_proba generated by inspect method on X and y
            complementary: flag that returns the cluster_id if set to False, otherwise the complementary dataframe (i.e.
                those with ~mask)

        Returns: tuple: Dataframe of sliced Shapley values, series of sliced targets, sliced probabilities
        """
        if self.cluster_report is None:
            self.compute()

        # Check if input specified by user, otherwise use the ones from self
        if summary_df is None:
            summary_df = self.summary_df
        if X_shap is None:
            X_shap = self.X_shap
        if y is None:
            y = self.y
        if predicted_proba is None:
            predicted_proba = self.predicted_proba

        mask = self.get_cluster_mask(summary_df, cluster_id)
        if not complementary:
            return X_shap[mask], y[mask], predicted_proba[mask]
        else:
            return X_shap[~mask], y[~mask], predicted_proba[~mask]

    def slice_cluster_eval_set(self, cluster_id, complementary=False):
        """
        Slices the input dataframes passed in the eval_set in the inspect function by the cluster id.

        Args:
            cluster_id: (int or list for multiple cluster_id) cluster ids to to slice
            complementary: flag that returns the cluster_id if set to False, otherwise the complementary dataframe (ie
                those with ~mask)

        Returns: list of tuplse: each element of the list containst
                    Dataframe of sliced shapley values, series of sliced targets, sliced probabilities
        """
        if not self.hasmultiple_dfs:
            raise NotFittedError("You did not fit the eval set. Please add an eval set when calling inspect()")

        output = []
        for X_shap, y, predicted_proba, summary_df in zip(
            self.X_shaps, self.ys, self.predicted_probas, self.summary_dfs
        ):
            output.append(
                self.slice_cluster(
                    cluster_id=cluster_id,
                    summary_df=summary_df,
                    X_shap=X_shap,
                    y=y,
                    predicted_proba=predicted_proba,
                    complementary=complementary,
                )
            )
        return output

    @staticmethod
    def get_cluster_mask(df, cluster_id):
        """
        Returns the mask to filter the cluster id.

        Args:
            df: dataframe with 'cluster_id' in it
            cluster_id: int or list of cluster ids to mask
        """
        if not isinstance(cluster_id, list):
            cluster_id = [cluster_id]

        mask = df["cluster_id"].isin(cluster_id)
        return mask

    @staticmethod
    def create_summary_df(cluster, y, probas, normalize=False):
        """
        Creates a summary.

        by concatenating the cluster series, the targets, the probabilities and the measured confusion.

        Args:
            cluster: pd.Series of clusters
            y: pd.Series of targets
            probas: pd.Series of predicted probabilities of the model
            normalize: boolean (if the predicted probabilities should be normalized to the max value

        Returns: pd.DataFrame (concatenation of the inputs)
        """
        confusion = return_confusion_metric(y, probas, normalize=normalize).rename("confusion")

        summary = [cluster, y.rename("target"), probas, confusion]

        return pd.concat(summary, axis=1)

    @staticmethod
    def aggregate_summary_df(df):
        """
        Performs the aggregations at the cluster_id level needed to generate the report of the inspection.

        Args:
            df: input df to aggregate

        Returns: pd.Dataframe with aggregation results
        """
        out = (
            df.groupby("cluster_id")
            .agg(
                total_label_1=pd.NamedAgg(column="target", aggfunc="sum"),
                total_entries=pd.NamedAgg(column="target", aggfunc="count"),
                label_1_rate=pd.NamedAgg(column="target", aggfunc="mean"),
                average_confusion=pd.NamedAgg(column="confusion", aggfunc="mean"),
                average_pred_proba=pd.NamedAgg(column="pred_proba", aggfunc="mean"),
            )
            .reset_index()
            .rename(columns={"index": "cluster_id"})
            .sort_values(by="cluster_id")
        )

        return out

    def fit_compute(self, X, y=None, eval_set=None, sample_names=None, **shap_kwargs):
        """
        Fits and orchestrates the cluster calculations and returns the computed report.

        Args:
            X: (pd.DataFrame) with the features set used to train the model
            y: (pd.Series, default=None): targets used to train the model
            eval_set: (list, default=None). list of tuples in the shape (X,y) containing evaluation samples, for example
                a test sample, validation sample etc... X corresponds to the feature set of the sample, y corresponds
                to the targets of the samples
            sample_names: (list of strings, default=None): list of suffixed for the samples. If none, it will be labelled with
                sample_{i}, where i corresponds to the index of the sample.
                List length must match that of eval_set
            **shap_kwargs:  kwargs to pass to the Shapley Tree Explained

        Returns:
            (pd.DataFrame) Report with aggregations described in compute() method.
        """  # noqa
        self.fit(X, y, eval_set, sample_names, **shap_kwargs)
        return self.compute()
