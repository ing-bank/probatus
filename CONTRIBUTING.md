# Contributing guide

`Probatus` aims to provide a set of tools that can speed up common workflows around validating regressors & classifiers and the data used to train them.
We're very much open to contributions but there are some things to keep in mind:

- Discuss the feature and implementation you want to add on Github before you write a PR for it. On disagreements, maintainer(s) will have the final word.
- Features need a somewhat general use case. If the use case is very niche it will be hard for us to consider maintaining it.
- If you’re going to add a feature, consider if you could help out in the maintenance of it.
- When issues or pull requests are not going to be resolved or merged, they should be closed as soon as possible. This is kinder than deciding this after a long period. Our issue tracker should reflect work to be done.

That said, there are many ways to contribute to Probatus, including:

- Contribution to code
- Improving the documentation
- Reviewing merge requests
- Investigating bugs
- Reporting issues

Starting out with open source? See the guide [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/) and have a look at [our issues labelled *good first issue*](https://github.com/ing-bank/probatus/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). GitHub Actions uses uv 0.12.7.
On macOS, install the OpenMP runtime for LightGBM with `brew install libomp`.

Run the same checks as GitHub Actions from the repository root:

```shell
uv run --locked --all-extras python scripts/check.py
```

This installs the environment from `uv.lock`, then runs all pre-commit hooks, the test suite
with coverage for the whole package, and the documentation build. Checks stop at the first failure.
Hooks can fix files; review their changes and rerun the command when that happens.
`--locked` rejects an outdated lockfile instead of silently changing dependencies.

To select checks or a Python version:

```shell
uv run --locked --all-extras python scripts/check.py --checks lint
uv run --locked --all-extras --python 3.13 python scripts/check.py
uv run --locked --all-extras python scripts/check.py --checks tests --notebooks
```

Notebook execution tests are skipped by default, just as in CI; `--notebooks` enables them.
Local checks use your operating system. GitHub additionally runs the same command across
Python 3.10–3.13 on Linux, macOS, and Windows. Coverage uploads and publishing remain GitHub steps.
The workflow definitions share `.github/workflows/checks.yml`; the actual validation commands
live in `scripts/check.py` so local checks and CI stay in sync.

To reproduce the weekly dependency-upgrade job:

```shell
uv lock --upgrade
uv run --locked --all-extras python scripts/check.py
```

This updates `uv.lock`; review the resulting dependency changes before committing them.
The minimum versions in `pyproject.toml` allow newer releases; the lockfile records the resolved versions for each supported Python version.

Python 3.10 uses SHAP 0.49.x and XGBoost 3.0.x: SHAP 0.50 dropped Python 3.10 support,
and older SHAP cannot load XGBoost 3.1's vector-valued base score.
Python 3.11 and newer use SHAP 0.50+ and XGBoost 3.2.x. XGBoost 3.3+ enables categorical
handling by default, which SHAP 0.52 rejects for interventional explanations even on numerical data.
Recheck these compatibility limits when upgrading SHAP.
See the [SHAP release notes](https://shap.readthedocs.io/en/stable/release_notes.html).

Install the Git hooks once for each clone:

```shell
uv run --locked --all-extras pre-commit install
```

This installs both pre-commit and pre-push hooks. Commits run the existing lint and type checks;
every push runs `scripts/check.py` with locked dependencies and is blocked if any check fails.
If a hook fixes files, review and commit the changes before pushing again.
The pre-push hook checks your local OS and Python version; GitHub still checks the full matrix.

Ruff handles linting, import sorting, Python syntax upgrades, and formatting for both Python files and notebooks:

```shell
uv run --all-extras ruff check --fix .
uv run --all-extras ruff format .
```

The development dependencies include notebook execution support. To edit notebooks in JupyterLab, run
`uv run --all-extras --with jupyterlab jupyter lab`.
Mypy remains the type checker; nbQA runs it on notebooks.

Probatus uses Python's standard `logging` module. To display informational messages from estimators with
`verbose=2`, configure logging in your application:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Standards

- Python 3.10–3.13
- Follow [PEP8](http://pep8.org/) as closely as possible (except line length)
- [google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/)
- Git: Include a short description of *what* and *why* was done, *how* can be seen in the code. Use present tense, imperative mood
- Git: limit the length of the first line to 72 chars. You can use multiple messages to specify a second (longer) line: `git commit -m "Patch load function" -m "This is a much longer explanation of what was done"`


### Code structure

* Model validation modules assume that trained models passed for validation are developed in a scikit-learn framework (i.e. have predict_proba and other standard functions), or follow a scikit-learn API e.g. XGBoost.
* Every python file used for model validation needs to be in `/probatus/`
* Class structure for a given module should have a base class and specific functionality classes that inherit from base. If a given module implements only a single way of computing the output, the base class is not required.
* Functions should not be as short as possible in terms of lines of code. If a lot of code is needed, try to put together snippets of code into other functions. This make the code more readable, and easier to test.
* Classes follow the probatus API structure:
    * Each class implements `fit()`, `compute()` and `fit_compute()` methods. `fit()` is used to fit an object with provided data (unless no fit is required), and `compute()` calculates the output e.g. DataFrame with a report for the user. Lastly, `fit_compute()` applies one after the other.
    * If applicable, the `plot()` method presents the user with the appropriate graphs.
    * For `compute()` and `plot()`, check if the object is fitted first.


### Documentation

Documentation is a very crucial part of the project because it ensures usability of the package. We develop the docs in the following way:

* We use [mkdocs](https://www.mkdocs.org/) with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) theme. The `docs/` folder contains all the relevant documentation.
* We use `mkdocs serve` to view the documentation locally. Use it to test the documentation everytime you make any changes.
* Maintainers can deploy the docs using `mkdocs gh-deploy`. The documentation is deployed to `https://ing-bank.github.io/probatus/`.
