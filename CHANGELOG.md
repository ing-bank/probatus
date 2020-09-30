# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
