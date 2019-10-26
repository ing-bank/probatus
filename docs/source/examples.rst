Examples
========

 1. `Binning <nb_binning.html>`_ notebook explains how the various implemented binning strategies of probatus work.Some of the implemented strategies are Simple binning, Quantile binning and binning by Algomerative clustering

 2. `Resemblance Models <nb_resemblance_modeling.html>`_ try to measure how different two samples are from a multivariate perspective. It works as follows:

    #. takes in input two datasets, X1, and X2
    #. it labels them with 0 and 1
    #. builds a model that will try to predict to which sample does an observation belong.
    #. when this model has an AUC = 0.5, it means that the samples are not distinguishable.

 3. `Distribution Statistics <nb_distribution_statistics.html>`_ helps to  check the stability of a feature in your model over time. Does the feature data from 2016-2018 that you used to train your model still describe your latest data from 2019, or has the feature population changed, requiring you to update your model ?

 4. `Probability Calibration <nb_calibration.html>`_ helps to calibrate the estimated probabilities .

 5. `Metric_Uncertainity <nb_metric_uncertainity.html>`_ helps to estimate the uncertainty around your chosen metric eg AUC.


Python notebooks for the above examples can be found here_ .

.. _here: https://gitlab.com/ing_rpaa/probatus/tree/master/notebooks