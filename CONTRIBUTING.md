# Contributing guide

`probatus` aims to provide a set of tools that can speed up common workflows around validating binary classifiers and the data used to train them.
We're very much open to contributions but there are some things to keep in mind:

- Discuss the feature and implementation you want to add on Github before you write a PR for it. On disagreements, maintainer(s) will have the final word.
- Features need a somewhat general usecase. If the usecase is very niche it will be hard for us to consider maintaining it.
- If you’re going to add a feature, consider if you could help out in the maintenance of it.
- When issues or pull requests are not going to be resolved or merged, they should be closed as soon as possible. This is kinder than deciding this after a long period. Our issue tracker should reflect work to be done.

That said, there are many ways to contribute to probatus, including:

- Contribution to code
- Improving the documentation
- Reviewing merge requests
- Investigating bugs
- Reporting issues

Starting out with open source? See the guide [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/) and have a look at [our issues labelled *good first issue*](https://github.com/ing-bank/probatus/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Setup

Development install:

```shell
pip install -e 'probatus[all]'
```

Unit testing:

```shell
pytest
```

We use [pre-commit](https://pre-commit.com/) hooks to ensure code styling. Install with:

```shell
pre-commit install
```

## Standards

- Python 3.6+
- Follow [PEP8](http://pep8.org/) as closely as possible (except line length)
- [google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/)
- Git: Include a short description of *what* and *why* was done, *how* can be seen in the code. Use present tense, imperative mood
- Git: limit the length of the first line to 72 chars. You can use multiple messages to specify a second (longer) line: `git commit -m "Patch load function" -m "This is a much longer explanation of what was done"`


### Code structure

* Model validation modules assume that trained models passed for validation are developed in a scikit-learn framework (i.e. have predict_proba and other standard functions), or follow a scikit-learn API e.g. XGBoost.
* Every python file used for model validation needs to be in `/probatus/`
* Class structure for a given module should have a base class and specific functionality classes that inherit from base. If a given module implements only a single way of computing the output, the base class is not required. 
* Functions should not be as short as possible in terms of lines of code. If a lot of code is needed, try to put together snippets of code into 
other functions. This make the code more readable, and easier to test.
* Classes follow the probatus API structure:
    * Each class implements `fit()`, `compute()` and `fit_compute()` methods. `fit()` is used to fit an object with provided data (unless no fit is required), and `compute()` calculates the output e.g. DataFrame with a report for the user. Lastly, `fit_compute()` applies one after the other.
    * If applicable, the `plot()` method presents the user with the appropriate graphs.
    * For `compute()` and `plot()`, check if the object is fitted first.
        

### Documentation

Documentation is a very crucial part of the project because it ensures usability of the package. We develop the docs in the following way:

* We use [mkdocs](https://www.mkdocs.org/) with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) theme. The `docs/` folder contains all the relevant documentation.
* We use `mkdocs serve` to view the documentation locally. Use it to test the documentation everytime you make any changes.
* Maintainers can deploy the docs using `mkdocs gh-deploy`. The documentation is deployed to `https://ing-bank.github.io/probatus/`.
