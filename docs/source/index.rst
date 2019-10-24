Welcome to Probatus' documentation!
====================================

.. image:: logo_large.png
  :alt: Probatus Logo

Library that standardizes and collects different validation steps to be performed on the models. 
We focus on binary classification models (most credit risk models)


Installation
************


Install `probatus` via pip with

.. code-block:: bash

   pip install probatus


Alternatively you can fork/clone and run:

.. code-block:: bash

    git clone https://gitlab.com/ing_rpaa/probatus.git
    cd probatus
    pip install .

Usage
*****

.. code-block:: python

    from probatus.binning import QuantileBucketer

    myQuantileBucketer = QuantileBucketer(bin_count=4)
    myQuantileBucketer.fit(x)
    print('counts', myQuantileBucketer.counts)
    print('boundaries', myQuantileBucketer.boundaries)

   ...

Examples
********

1.  `Binning <nb_binning.html>`_ notebook explains how the various implemented binning strategies of probatus work.
    Some of the implemented strategies are Simple binning, Quantile binning and binning by Algomerative clustering


2. `Resemblance Models <nb_resemblance_modeling.html>`_ try to measure how different two samples are from a multivariate perspective.

    It works as follows:

    #. takes in input two datasets, X1, and X2
    #. it labels them with 0 and 1
    #. builds a model that will try to predict to which sample does an observation belong.
    #. when this model has an AUC = 0.5, it means that the samples are not distinguishable.

3. `Distribution Statistics <nb_distribution_statistics.html>`_ helps to  check the stability of a feature in your
model over time. Does the feature data from 2016-2018 that you used to train your model still describe your latest
data from 2019, or has the feature population changed, requiring you to update your model?


More examples can be found here_ .

.. _here: https://gitlab.com/ing_rpaa/probatus/tree/master/notebooks


Contribute
**********

You can contribute to this code through Pull Request on GitLab_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitLab: https://gitlab.com/ing_rpaa/probatus.git


License
*******

.. toctree::
   :maxdepth: 2

   license
   nb_shap_importance.ipynb
   nb_binning.ipynb
   nb_calibration.ipynb
   nb_distribution_statistics.ipynb
   nb_metric_uncertainity.ipynb
   nb_resemblance_modeling.ipynb

Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
