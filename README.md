[![pipeline status](https://gitlab.com/ing_rpaa/probatus/badges/master/pipeline.svg)](https://gitlab.com/ing_rpaa/probatus/-/commits/master)
[![coverage report](https://gitlab.com/ing_rpaa/probatus/badges/master/coverage.svg)](https://gitlab.com/ing_rpaa/probatus/-/commits/master)
<img style="float: right;" alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/probatus">
![PyPI](https://img.shields.io/pypi/v/probatus)
![PyPI - Downloads](https://img.shields.io/pypi/dm/probatus)

# Probatus

## Overview

Library for the validation of binary classifiers

## Installation

```bash
pip install probatus
```

## Documentation

Latest documentation can be found [here](https://probatus.readthedocs.io/en/latest/)

## Features 

#### 1 Metrics for distirbution similarity check
Everything related to metrices used for the validation of the models.
On top of the usual metrics defined in sklearn, provide:
- [ ] PSI score
- [ ] KS values
- [ ] Hypothesis testing for distribution similarity
- [ ] **Something else the smart data scientists will think about ;)** 

#### 2 Metrics uncertainty
This part is often underestimated.
With unbalanced dataset and few cases of low there is a big issue as some.
- [ ] Resampling n_times (at least 100) of the train test split to get distirbution of the metrices (example AUC)
- [ ] Implement it in a parallel fashion (n_jobs) to speed it up
- [ ] **Smart way to correct for correlated indipendent folds**


#### 3 Bucketing strategy
Some metrices require different binning algorithms to be computed (for example, PSI).
This epic covers multiple algorithms related to the way how customers are grouped together used for risk bucketing calculations
- [ ] Implement simple binning
- [ ] Implement one dimensional clustering based on model prediction
- [ ] Implement the "Optyparm" algorithm
- [ ] **Think of different bucketing algorithms**


#### 4 Model calibration
Go beyond the isotonic regression for calibration - implement likelihood estimates with parametrizable functionality
- [ ] Implement fitting of "logistic regression" with non linear dependencies between the expected predicitons and the actual defatul rates per risk bucket

#### 5 Sample validation
Checks related to the validation of the samples definition. This includes building propensity models for the sample definiton
- [ ] Build different propensity models to validate the similarity of the samples
- [ ] Perform univarite checks of feature distributions over different samples
- [ ] Correlations change between samples

#### 6 Effects of correlations between features
- [ ] **Think about it**

#### 7 Time stability
Evaluate the stability of the model predictions over time
- [ ] Build different propensity models to validate the similarity of time samples 
- [ ] Perform univarite checks of feature distributions over different samples
- [ ] Calculate Shapley values over different time intervals and try to compare distirbutions with metrices define in epic 1
- [ ] Evaluate the stability of the model predictions in differnet time subsamples with metrices defined in epic 1 (to be used as model performance metric)
- [ ] **Something else?**

#### 8 Interpretability
Basically shapely values
- [ ] **Extend this**

#### 9 Features impact on models
This includes checks like permutation importance
- [ ] Implement permutation importance
- [ ] Define a strategy to estimate the systematic uncertainty of the feature (to be used for monitoring model performance)

#### 10 Model version control
Implement the Crayon way of autotagging the models if developed in a git repo
- [ ] Implement the current version control mechanism from Crayon

#### 11 Reporting
Make a simple reporting tool that generates a standrazie document with the most important findings
- [ ] Think about this - to be refined

#### 12 Bias detection
This one is hard
- [ ] **Think about this - to be refined abd rethought**

#### 13 Host it on a server
Make a flask app that enables a server like hosting (something like jupyter)
- [ ] Build a simple Flask Server
- [ ] Implement a command line interace (CLI) to run the flask app and lounche a web app (jupyter like)

#### 14 Front end
Related to the previous story, make a front end (maybe in Shiny?) to actually allow the modelers/validators/business people without any pyton knowledge to  use the capabilities of the library (kind of import your data, your model and get everything in a clickable UI)

#### 15 Define some criteria for model quality
Define some standard criteria for model quality that will give a quality score based on all the checks developed above

### 16 Feature interactions 
Factorization machines based approach - polynomial factorization regression

### Sample dataset is provided by the library

Sample test loan default data from Lending Club https://www.lendingclub.com/. Only a sample of all loans and available features is provided.

Metadata:

* id                         object - Loan ID
* loan_issue_date    datetime64[ns] - Date when loan was issued. All dates are set to fist day of the month
* default                     int64 - Flag if the loan went to default or not. 
                                    Default is defined as one of statuses: 'Late (31-120 days)', 'Late (16-30 days)', 'Default'
* loan_amnt                 float64 - Amount requested
* funded_amnt               float64 - Amount granted
* term                        int64 - Term of the loan in months
* int_rate                  float64 - Annual interest rate
* annual_inc                float64 - Annual income of the client
* fico_range_low            float64 - Fico rate range low
* fico_range_high           float64 - Fico rate range high
* long_emp                    int64 - Indicator if client is employed for longer than 5 years
* credit_grade_A              uint8 - Indicator if clients credit grade is A
* credit_grade_B              uint8 - Indicator if clients credit grade is B
* credit_grade_C              uint8 - Indicator if clients credit grade is C
* credit_grade_D              uint8 - Indicator if clients credit grade is D
* credit_grade_E              uint8 - Indicator if clients credit grade is E
* credit_grade_F              uint8 - Indicator if clients credit grade is F
* credit_grade_G              uint8 - Indicator if clients credit grade is G
