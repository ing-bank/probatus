# Sample Similarity

The goal of sample similarity module is understanding how different two samples are from a multivariate perspective.

One of the ways to indicate this is Resemblance Model. Having two datasets -  say X1 and X2 - one can analyse how easy it is to recognize which dataset a randomly selected row comes from. The Resemblance model assigns label 0 to the dataset X1, and label 1 to X2 and trains a binary classification model to predict which sample a given row comes from.
By looking at the test AUC, one can conclude that the samples have a different distribution if the AUC is significantly higher than 0.5. Furthermore, by analysing feature importance one can understand which of the features have predictive power.

<img src="../img/resemblance_model_schema.png"/>


The following features are implemented:

- [SHAPImportanceResemblance (Recommended)][probatus.sample_similarity.resemblance_model.SHAPImportanceResemblance]:
  The class applies SHAP library, in order to interpret the tree based resemblance model.
- [PermutationImportanceResemblance][probatus.sample_similarity.resemblance_model.PermutationImportanceResemblance]:
  The class applies permutation feature importance in order to understand which features the current model relies on the most. The higher the importance of the feature, the more a given feature possibly differs in X2 compared to X1. The importance indicates how much the test AUC drops if a given feature is permuted.


::: probatus.sample_similarity.resemblance_model

