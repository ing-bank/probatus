Welcome to Probatus' documentation!
====================================
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

More examples can be found here_ .

.. _here: https://gitlab.com/ing_rpaa/probatus/tree/master/notebooks


Contribute
**********

You can contribute to this code through Pull Request on GitLab_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitLab: https://gitlab.com/ing_rpaa/probatus.git


Contact
*******

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   license
   contact

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
