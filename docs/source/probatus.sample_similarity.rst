probatus.sample_similarity module
========================

The goal of sample similarity module is understanding how different two samples are from a multivariate perspective.

One of the ways to indicate that is Resemblance Model. Having two datasets say X1 and X2, one can analyse how easy is it to recognize which dataset a randomly selected row comes from. The Resemblance model assigns label 0 to X1 dataset, and label 1 to X2 and trains a binary classification model that to predict, which sample a given row comes from.
By looking at the test AUC, one can conclude that the samples have different distribution the AUC is significantly higher than 0.5. Further, by analysing feature importance one can understand, which of the features have predictive power.

The following features are implemented:

- **BaseResemblance** - Base class, provides main functionality with fit method that can be overwritten by subclasses.

   - **SHAPImportanceResemblance (Recommended)** -  The class applies SHAP library, in order to interpret the tree based resemblance model model.

   - **PermutationImportanceResemblance** -  The class applies permutation feature importance, in order to understand, which features does the current model rely the most on. The higher the importance of the feature, the more a given feature possibly differs in X2 compared to X1. The importance indicates how much the test AUC drops if a given feature is permuted.


probatus.sample_similarity module
-----------------------------

.. automodule:: probatus.sample_similarity.resemblance_model
    :members:
    :undoc-members:
    :show-inheritance:
