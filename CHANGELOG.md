# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.8] - 2021-12-08
Improvements in this release:
- Added support for XGBoost and Catboost models in ShapRFECV  #175

## [1.8.7] - 2021-10-28
Improvements in this release:
- Added support for early stopping in new lightgbm version #164

## [1.8.6] - 2021-10-05
Improvements in this release:
- Added alpha parameter to DependencePlotter #162

## [1.8.5] - 2021-08-24
Improvements in this release:
- Docs and docstrings improvements for stats tests #158

## [1.8.4] - 2021-06-16
Improvements in this release:
- Fix the bug in the Shap Dependence Plot #153
- Add HowTo guide for using grouped data #154

## [1.8.3] - 2021-06-15
Improvements in this release:
- Fix p-value calculation in PSI #142

## [1.8.2] - 2021-05-04
Improvements in this release:
- Fix catboost bug when calculating SHAP values #147
- Supply eval_sample_weight for fit in EarlyStoppingShapRFECV #144
- Remove codecov.io #145
- Remove sample_row from probatus #140

## [1.8.1] - 2021-04-18
Improvements in this release:
- Enable use of sample_weight in ShapRFECV and EarlyStoppingShapRFECV #139
- Fix bug in EarlyStoppingShapRFECV #139
- Fix issue with categorical features in SHAP #138
- Missing values handled by AutoDist #126
- Fix issue with missing histogram in DependencePlot #137
  
## [1.8.0] - 2021-04-14
Improvements in this release:
- Implemented EarlyStoppingShapRFECV #108
- Added support for Python 3.9 #132

## [1.7.1] - 2021-04-13
Improvements in this release:
- Add error if model pipeline passed to SHAP #129
- Fixed PSI bug with empty bins #116
- Unit tests are run daily #113
- TreeBucketer has been refactored #124
- Fixes to failing test pipeline #120
- Improving language in docs #109, #107 

## [1.7.0] - 2021-03-16
Improvements in this release:
- Create a comparison of imputation strategies #86
- Added support for passing check_additivity argument #103
- Range of code styling issues fixed, based on precommit config #100 
- Renamed TreeDependencePlotter to DependencePlotter and exposed the docs #94
- Enable instalation of extra dependencies #97
- Added how to notebook to ensure reproducibility #99
- Description of vision of probatus #91

## [1.6.2] - 2021-03-10
Improvements in this release:
- Bugfix, allow passing kwargs to dependence plot in ShapModelInterpreter #90

## [1.6.1] - 2021-03-09
Improvements in this release:
- Added ShapRFECV support for all sklearn compatible search CVs. #76 #49

## [1.6.0] - 2021-03-01
Improvements in this release:
- Added features list to README #53
- Added docs for sample row functionality #54
- Added 'open in colab' badges to tutorial notebooks #56
- Deploy documentation on release #47
- Added columns_to_keep for shap feature elimination #63
- Updated docs for usage of columns to keep functionality in SHAPRFECV #66
- Added shap support for linear models #69
- Installed probatus in colab notebooks #80
- Minor infrastructure tweaks #81

## [1.5.1] - 2020-12-04

Various improvements to the consistency and usability of the package
- Unit test docstring and notebooks #41 
- Unified scoring metric within probatus #27 
- Improve docstrings consistency documentation #25 
- Implemented unified interface #24 
- Added images to API docs documentation #23
- Added verbose parameter to ShapRFECV #21
- Make API more consistent #19 
    - Set model parameter name to clf across probatus
    - Set default random_state to None
    - Ensure that verbose is used consistently in probatus
    - Unify parameter class_names for classes in which it is relevant
    - Add return scores parameter to compute wherever applicable
- Add sample row functionality to utils #17
- Make an experiment comparing sklearn.RFECV with ShapRFECV #16
- ShapModelInterpreter calculate train set feature importance #13

## [1.5.0] - 2020-11-18
- Improve SHAP RFECV API and documentation

## [1.4.4] - 2020-11-11
- Fix issue with the distribution uploaded to pypi

## [1.4.0] - 2020-11-10 (Broken)
- Add SHAP RFECV for features elimination

## [1.3.0] - 2020-11-05 (Broken)
- Add SHAP Model Inspector with docs and tests

## [1.2.0] - 2020-09-30
- Add resemblance model, with SHAP based importance
- Improve the docs for resemblance model
- Refactor stats tests, improve docs and expose functionality to users

## [1.1.1] - 2020-09-08
- Improve Tree Bucketer, enable user to pass own tree object

## [1.1.0] - 2020-08-24
- Improve docs for stats_tests
- Refactor stats_tests

## [1.0.1] - 2020-08-07
- TreeBucketer, which bins the data based on the target distribution, using Decision Trees fitted on a single feature
- PSI calculation includes the p-values calculation

## [1.0.0] - 2020-02-24
- metric_volatility and sample_similarity rebuilt
- New documentation
- Faster tests
- Improved and simplified API
- Scorer class added to the package
- Removed data from repository
- Hiding unfinished functionality from the user

## [0.1.3] - 2020-02-24

### Added

- VolalityEstimation now has random_seed argument

### Changed

- Improved unit testing
- Improved documentation README and CONTRIBUTING

### Fixed

- Added dependency on scipy 1.4+

## [0.1.2] - 2019-10-29
### Added

- Readthedocs documentation website

## [0.1.1] - 2019-10-09

### Added

- Added CHANGELOG.md

### Changed 

- Renamed to probatus
- Improved testing by adding pyflakes to CI
- probatus.metric_uncertainty.VolatilityEstimation is now deterministic, added random_state parameter 

## [0.1.0] - 2019-09-21

Initial release, commit ecbd0d08a6eea370afda4a4790edeb4ee382995c

[Unreleased]: https://gitlab.com/ing_rpaa/probatus/compare/ecbd0d08a6eea370afda4a4790edeb4ee382995c...master
[0.1.0]: https://gitlab.com/ing_rpaa/probatus/commit/ecbd0d08a6eea370afda4a4790edeb4ee382995c
