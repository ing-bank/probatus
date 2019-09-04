
from ..utils import NotFittedError, UnsupportedModelError
import numpy as np
import pandas as pd

#import hdbscan
from sklearn.cluster import KMeans
from ._shap_helpers import shap_to_df


def return_confusion_metric(y_true, y_score, normalize = False):
    """
    Computes a confusion metric as absolute difference between the y_true and y_score.
    If normalize eis set to tru, it will normalize y_score to the maximum value in the array
    Args:
        y_true: (np.ndarray) true targets
        y_score: (np.ndarray) model output
        normalize: boolean, normalize or not to the maximum vlaue

    Returns: (np.ndarray) conflusion metric

    """

    if normalize:
        y_score = y_score/y_score.max()

    return np.abs(y_true - y_score)

def check_is_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise NotImplementedError("Sorry, X needs to be a pd.DataFrame for now")


class BaseInspector(object):
    def __init__(self, algotype, **kwargs):
        self.fitted = False
        self.algotype = algotype
        # TODO fix compilatiopn issue on  for hdbscan
        # if algotype =='dbscan':
        #     self.clusterer = hdbscan.HDBSCAN(prediction_data=True,**kwargs)
        if algotype =='kmeans':
            self.clusterer = KMeans(**kwargs)
        else:
            raise UnsupportedModelError("The algorithm {} is not supported".format(algotype))

    def __repr__(self):
        repr_ = "{},\n\t{}".format(self.__class__.__name__,self.algotype)
        if self.fitted:
            repr_ += "\n\tTotal clusters {}".format(np.unique(self.clusterer.labels_).shape[0])
        return repr_

    def fit_clusters(self, X):
        """
        Perform the fit of the clusters with the algorithm specified in the constructor
        Args:
            X: input features

        Returns: cluster labels

        """
        self.clusterer.fit(X)
        self.fitted = True

        return self

    def predict_clusters(self,X):
        if not self.fitted:
            raise NotFittedError("Inspector not fitter. Run .fit()")

        labels = None
        if self.algotype == 'kmeans':
            labels =  self.clusterer.predict(X)
        if self.algotype == 'dbscan':
            labels, strengths = hdbscan.approximate_predict(self.clusterer, X)

        return labels

    @staticmethod
    def check_is_dataframe(df):

        if isinstance(df,pd.DataFrame):
            return df

        elif isinstance(df,np.ndarray) and len(df.shape)==2:
            return pd.DataFrame(df)

        else:
            raise NotImplementedError("Sorry, X needs to be a pd.DataFrame for for a 2 dimensional numpy array")

    @staticmethod
    def assert_is_series(series, index=None):

        if isinstance(series, pd.Series):
            return series
        elif isinstance(series, pd.DataFrame) and series.shape[1] == 1:
            return pd.Series(series.values.ravel(), index=series.index)
        elif isinstance(series, np.ndarray) and index is not None:
            return pd.Series(series, index=index)
        else:
            raise TypeError(
                "The object should be a pd.Series, a dataframe with one collumn or a 1 dimensional numpy array")




class InspectorShap(BaseInspector):
    """
    Class to perform inspection of the model prediction based on Shapley values.

    It uses the calculated shapley values for the train model to build clusters in the shap space.
    For each cluster, an average confusion, average predicted probability and observed rate of a single class is
    calculated.
    Every sub cluster can be retrieved with the function slice_cluster to perform deeper analysis.

    The original dataframe indexing is used in slicing the dataframe, ensuring easy filtering

    Args:
            model: (obj) pretrained model (with sklearn-like API)
            algotype: (str) clustering algorithm (supported are kmeans and hdbscan)
            confusion_metric: (str) Confusion metric to use:
                - "proba": it will calculate the confusion metric as the absolute value of the target minus the predicted
                           probability. This provides a continuous measure od confusion, where 0 indicated correct predictions
                           and the closer the number is to 1, the higher the confusion
            normalize_probability: (boolean) if true, it will normalize the probabilities to the max value when computing
                the confusion metric
            cluster_probabilities: (boolean) if tru, uses the model prediction as an input for the cluster prediction
            **kwargs: keyword arguments for the clustering algorithm

    """


    def __init__(self, model, algotype='kmeans', confusion_metric = 'proba',
                 normalize_probability=False,cluster_probability = False, **kwargs):

        super().__init__(algotype, **kwargs)
        self.model = model
        self.isinspected = False
        self.hasmultiple_dfs = False
        self.normalize_proba = normalize_probability
        self.cluster_probabilities = cluster_probability

        if confusion_metric not in ['proba']:
            #TODO implement the target method
            raise NotImplementedError("confusion metric {} not supported. See docstrings".format(confusion_metric))
        self.confusion_metric = confusion_metric
        self.cluster_report = None

    def __repr__(self):
        repr_ = "{},\n\t{}".format(self.__class__.__name__, self.algotype)
        if self.fitted:
            repr_ += "\n\tTotal clusters {}".format(np.unique(self.clusterer.labels_).shape[0])
        return repr_


    def compute_probabilities(self,X):
        """
        Compute the probabilities for the model using the sklearn API
        Args:
            X: Feature set

        Returns: (np.array) probability

        """

        return self.model.predict_proba(X)[:,1]


    def fit_clusters(self, X):
        """
        Perform the fit of the clusters with the algorithm specified in the constructor
        Args:
            X: input features


        """

        X = X.copy()
        if self.cluster_probabilities:
            X['probs'] = self.predict_proba

        return super().fit_clusters(X)


    def predict_clusters(self,X):
        """
        Predicts the clusters of the dataset X
        Args:
            X: features

        Returns: cluster labels

        """
        X = X.copy()
        if self.cluster_probabilities:
            X['probs'] = self.predict_proba

        return super().predict_clusters(X)

    def inspect(self, X, y=None, eval_set = None, sample_names=None, **shap_kwargs):
        """
        Performs the cluster calculations
        Args:
            X: pd.DataFrame with the features set used to train the model
            y: pd.Series (default None): targets used to train the model
            eval_set: (list, default None). list of tuples in the shape (X,y) containing evaluation samples, for example
                a test sample, validation sample etc... X corresponds to the feature set of the sample, y corresponds
                to the targets of the samples
            sample_names: (list of strings): list of suffixed for the samples. If none, it will be labelled with
                sample_{i}, where i corresponds to the index of the sample.
                List length must match that of eval_set
            **shap_kwargs:  kwargs to pass to the Shapley Tree Explained

        """

        X = self.check_is_dataframe(X)
        y = self.assert_is_series(y, index = X.index)

        self.set_names = sample_names
        if sample_names is not None:
            # Make sure that the amount of eval sets matches the set names
            assert len(eval_set) == len(sample_names), "set_names must be the same length as eval_set"
            self.set_names = sample_names


        self.X_shap = shap_to_df(self.model, X, **shap_kwargs)
        self.y = y
        self.predict_proba = pd.Series(self.compute_probabilities(X), index = self.y.index,name = 'pred_proba')
        self.fit_clusters(self.X_shap)
        self.clusters = pd.Series(self.predict_clusters(self.X_shap),index = self.y.index, name= 'cluster_id')

        if eval_set is not None:

            assert isinstance(eval_set, list), "eval_set needs to be a list"

            self.hasmultiple_dfs = True
            self.X_shaps = list()
            self.clusters_list = list()
            self.ys = list()
            self.predict_probas = list()
            for X_, y_ in eval_set:
                X_ = self.check_is_dataframe(X_)
                y_ = self.assert_is_series(y_, index=X_.index)

                X_shap_ = shap_to_df(self.model, X_, **shap_kwargs)
                self.X_shaps.append(X_shap_)
                self.ys.append(y_)
                self.predict_probas.append(pd.Series(self.compute_probabilities(X_), index = y_.index, name = 'pred_proba'))
                clusters_ = pd.Series(self.predict_clusters(X_shap_),index = y_.index,  name= 'cluster_id')
                self.clusters_list.append(clusters_)

        return self

    def _compute_report(self):
        """
        Helper function to compute the report of the ispector - performs aggregations per cluster id


        """

        self.summary_df = self.create_summary_df(self.clusters, self.y, self.predict_proba, normalize=self.normalize_proba)
        self.agg_summary_df = self.aggregate_summary_df(self.summary_df)

        if self.hasmultiple_dfs:

            self.summary_dfs = [
                self.create_summary_df(clust, y, pred_proba, normalize=self.normalize_proba)
                for  clust, y, pred_proba in zip(self.clusters_list, self.ys, self.predict_probas)
            ]

            self.agg_summary_dfs = [
                self.aggregate_summary_df(df)
                for df in self.summary_dfs
            ]


    def get_report(self):
        """
        Calculates a report containing the information per cluster.
        Includes the following:
            - cluster id
            - total number of observations in the cluster
            - total number of target 1 in the cluster
            - target 1 rate (ration of target 1 counts/observations)
            - average predicted probabilitites
            - average confusion

        If multiple eval_sets were passed in the inspect() functions, the output will contain those aggregations as well
        The output names will use the sample names provided in the inspect function. Otherwise they will be labelled by
        the suffix sample_{i}, where i is the index of the sample

        Returns: (pd.DataFrame) with above mentioned aggregations.

        """

        if self.cluster_report is not None:
            return self.cluster_report

        self._compute_report()
        out = self.agg_summary_df.copy()


        if self.hasmultiple_dfs:

            for ix, agg_summary_df in enumerate(self.agg_summary_dfs):
                if self.set_names is None:
                    sample_suffix = "sample_{}".format(ix+1)
                else: sample_suffix = self.set_names[ix]

                out = pd.merge(out,agg_summary_df, on='cluster_id',  suffixes = ('','_{}'.format(sample_suffix)))


        self.cluster_report = out
        return self.cluster_report



    def slice_cluster(self, cluster_id, complementary = False):
        """
        Slices the input dataframes by the cluster.

        Args:
            cluster_id: (int or list for multiple cluster_id) cluster ids to to slice
            complementary: flag that returns the cluster_id if set to False, otherwise the complementary dataframe (ie
                those with ~mask)

        Returns: tuple: Dataframe of sliced shapley values, series of sliced targets, sliced probabilities

        """

        if self.cluster_report is None:
            self.get_report()

        mask = self.get_cluster_mask(self.summary_df, cluster_id)
        if not complementary:
            return self.X_shap[mask], self.y[mask], self.predict_proba[mask]
        else:
            return self.X_shap[~mask], self.y[~mask], self.predict_proba[~mask]


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

        if self.cluster_report is None:
            self.get_report()


        output = list()
        for X_shap, y, predict_proba, summary_df in zip(self.X_shaps, self.ys, self.predict_probas, self.summary_dfs):
            mask = self.get_cluster_mask(summary_df, cluster_id)
            if not complementary:
                output.append((X_shap[mask], y[mask], predict_proba[mask]))
            else:
                output.append((X_shap[~mask],y[~mask], predict_proba[~mask]))

        return output



    @staticmethod
    def get_cluster_mask(df, cluster_id):
        """
        Returns the mask to filter the cluster id
        Args:
            df: dataframe with 'cluster_id' in it
            cluster_id: int or list of cluster ids to mask

        Returns:

        """

        if not isinstance(cluster_id, list):
            cluster_id = [cluster_id]

        mask = df['cluster_id'].isin(cluster_id)
        return mask

    @staticmethod
    def create_summary_df(cluster,y, probas, normalize=False):
        """
        Creates a summary by concatenating the cluster series, the targets, the probabilities and the measured confusion
        Args:
            cluster: pd.Series od clusters
            y: pd.Series od targets
            probas: pd.Series of predicted probabilities of the model
            normalize: boolean (if the predicted probabilities should be normalized to the max value

        Returns: pd.DataFrame (concatenation of the inputs)

        """

        confusion = return_confusion_metric(y,probas, normalize = normalize).rename("confusion")

        summary = [
            cluster,
            y.rename("target"),
            probas,
            confusion
        ]

        return pd.concat(summary, axis=1)

    @staticmethod
    def aggregate_summary_df(df):
        """
        Performs the aggregations at the cluster_id level needed to generate the report of the inspection
        Args:
            df: input df to aggregate

        Returns: pd.Dataframe with aggregation results
        """


        out = df.groupby("cluster_id", as_index=False).agg(
            total_label_1=pd.NamedAgg(column='target', aggfunc="sum"),
            total_entries=pd.NamedAgg(column='target', aggfunc="count"),
            label_1_rate=pd.NamedAgg(column='target', aggfunc="mean"),

            average_confusion=pd.NamedAgg(column='confusion', aggfunc="mean"),


            average_pred_proba=pd.NamedAgg(column='pred_proba', aggfunc="mean"),

        ).reset_index().rename(columns={"index": "cluster_id"}).sort_values(by='cluster_id')

        return out


