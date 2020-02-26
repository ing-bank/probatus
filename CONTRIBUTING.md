# Contributing guide

Work at ING? Join our slack channel `#probatus`! :coffee:

There are many ways to contribute to probatus, including:

- Contribution to code
- Improving the documentation
- Reviewing merge requests
- Investigating bugs
- Reporting issues

In this guide, we describe the best practices to contribute a project.

### Table of Contents
<!---
To update the table of contents after a change, you can use the markdown-toc npm package: `markdown-toc -i CONTRIBUTING.md`
https://github.com/jonschlinkert/markdown-toc
--->

<!-- toc -->


- [Code development](#code-development)
  * [Jupyter notebooks](#jupyter-notebooks)
  * [Functions in .py files](#functions-in-.py-files)

- [Technical Standards](#technical-standards)
  * [Python](#python)
  * [Jupyter notebooks](#jupyter-notebooks)
  * [Code for model validation modules](#code-for-model-validation-modules)
  * [Functions for multiple modules](#functions-for-multiple-modules)
  
- [Project Data](#project-data)
- [Code Reviews](#code-reviews)
  * [What to review](#what-to-review)
  * [Code review workflows](#code-review-workflows)

- [Using GIT](#using-git)
  * [Development workflow with GIT: Branching strategy](#development-workflow-with-git-branching-strategy)
  * [Commit Messages](#optional-commit-messages)

<!-- tocstop -->



## Code development

### Jupyter notebooks

- Do not collaborate on notebooks. Always create your own personal notebooks. This is because they are hard to version control as they include more than just code (cells & output).
- **Clear output cells that may contain sensitive information before commits to prevent personal data leaking into the git repo.**
- Try to clear large cell outputs before committing. Printing huge tables or plotting large graphs will only bloat your git repository and you run a larget risk of leaking sensitive data.
- Do not copy and paste code. Instead write reusable functions in `src/` and load them into your notebooks (tip: use `%autoreload`). This blogpost details the workflow: [write less terrible notebook code](https://blog.godatadriven.com/write-less-terrible-notebook-code)


### Functions in .py files

- Reusable functions must be stored into `.py files`. This includes (but is not limited to):
    * data loading
    * data cleaning
    * feature generation
    * model hyperparameter tuning
    * plotting
    * validation steps
- The `.py files` must be stored into the directory `/probatus/` or a subdirectory of the same
- Written functions should follow the [Technical Standards](#technical-standards)
- Verbosity
    * There should be no print statements to stdout by default.
    * If you want optional printing to stdout, add the argument `verbose` to your function, by default set to `False`.



## Technical Standards

### Python

* Use Python 3. The point-version (3.4/3.5/etc.) depends on availability in your dev/prod environment.
* Use environments (conda). To create one, run `conda env create -f requirements.yml`.
* Keep track of dependencies in `requirements.yml` and run `conda install --file requirements.yml` to install them. Remember to notify your colleagues of the change in requirements since they'll need to conda install them.
* Use environment variables for simple configurations and secrets
  * You can load the environment variables in Python scripts using the [https://github.com/theskumar/python-dotenv](dotenv) package.
  * You can load the variables in bash scripts using the `dotenv.sh` script
* Line length is 120
* Follow [PEP8](http://pep8.org/) as closely as possible (except line length)
* Use ```__name__ == '__main__'```, otherwise Sphinx will execute your script upon building.
* Use the [google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/)

```python
def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    return True
```

### Code for model validation modules
* Model validation modules assume that trained models passed for validation are developed in scikit-learn framework (have predict_proba and other standard functions)
* They implement fit and transform methods and preferably inherit from BaseEstimator and TransformerMixin
* Every python file used for model validation, needs to be in `/probatus/`
* Every python module that validate the model needs to take at minimum three inputs: 
    - model - model trained in scikit-learn framework
    - credit_data - pandas DataFrame with features and targets
    - X - array/list with names of featuers
    - y - string with name of the target variables
    - Other necessary parameters are allowed, but they need to be predefined
* An example in pseudocode of a function generating the features is as follows:

```python 
def KS_test(model, credit_data, X, Y):
    """
    Description of the function
    """
    # does operations with the tables and returns the feature
    
    return ks__scores
```
* Functions should not exceed 10-15 lines of code. If more code is needed, try to put together snippets of code into 
other functions. This make the code more readable, and easier to test.

### Functions for multiple modules

Some functions might be useful for multiple features. Therefore, it does not make sense to store them in 
`feature_specific.py` file, but rather create add them in a helper file, i.e. `src/features/helpers.py`.<br>
An example:
- A function that helps with repartitioning the tables could be defined in `helpers.py`
- A function that helps with the naming convention (for example with the time interval definiton) could be
 defined in `helpers.py`
 - A function converts all IDs from String to Integers could be defined in `helpers.py`

    

### Line endings (optional)

Windows vs Linux line endings can be a huge source of trouble.
We have included a `.gitattributes` file that forces unix-style line endings (LF).

If you do run into trouble, please see our [dealing with windows line endings guide](https://confluence.europe.intranet/display/IAAT/Dealing+with+windows+line+endings+in+GIT).

## Project Data

- **Do not commit data to git!**. As git preserves history, onces data has been pushed to gitlab it is **very** difficult to remove.
- Do not add pickle models into the repo, since they are dependent on the sklearn version used to initialize them
- Very small .csv/.pkl files can be permitted when used for examples, testing, metadata or configuration purposes
- Source data should never be edited, instead create a reproducible pipeline with the immutable source data as input
- Large files should go into HDFS, preferably in the parquet format or stored in Hive, depending on your development environment
- Use the `/data` folder to store project data. This is .gitignore-ed directory by default

## Code Reviews

Code reviews are very important: 
- improve knowledge of both reviewer and reviewee
- improve consistency of code, easier to work on in the future
- shorter development time as you catch bugs earlier
- smaller chance of bugs in production
- "*Code review is about knowledge sharing, not defect detection. As soon as you accept this and embrace static analysis your quality will go up dramatically and code reviews will become more useful.*" - [Riley Berton Tweet](https://twitter.com/rileyberton/status/1018586513142091781)

Even when you don't expect your code to go into production, you can benefit from code reviews! 

### What to review

In a code review, you look for a number of things:

- Correctness
    - Does the work solve the problem outlined in the JIRA story/ Gitlab issue?
    - Does it run?
    - Does it do what it promises?
    - Joins: are the number of rows what you expect?
    - Are the number of N.A.s reasonable?
- Clarity
    - Is it immediately clear what this code does?
- Documentation
    - Are there docstrings in the agreed format?
    - Is every module & script documented?
    - If necessary, is there a demo notebook?
- Hygiene
    - Does it adhere to the team's coding standards?
    - Is the style consistent?
    - Does every function do a single thing?
    - If relevant: Does main.py run and include the new code?
    - (depending on team's style) Finished code is moved to python modules and re-used in notebooks (see blogpost [write-less-terrible-notebook-code](https://blog.godatadriven.com/write-less-terrible-notebook-code)).
- Running time: does it finish within a reasonable time - but don't optimize prematurely!
- Tests (optional, if it makes sense and if it makes the review easier)
- Education: Is there an alternative solution? You can name it even if you don't reject the review because of it.

It is the responsibility of the reviewee to make sure that the code is easy to review. Some more background: 
- [Gitlab's code review guidelines](https://docs.gitlab.com/ee/development/code_review.html) has some good bullets on how to do code reviews.
- [thoughtbot code review guide](https://github.com/thoughtbot/guides/tree/master/code-review)

## Using GIT

### Development workflow with GIT: Branching strategy

**Development on feature branches, merge to master**:
- Develop the code in the branch created for the issue
- Once the code is ready to be merged with master, resolve WIP status and create a new pull request
- Ask one of your peers to review the code and merge the pull request
- Any feature added to the project should include unit tests. If anyone modifies the features that are already there, tests should be refined accordingly. Finally if a bug is found, a test should be written that makes sure this bug would be detected in the future
    
<!---
- **Development on feature branches**:
    - Create a git issue for every chunck of work you do
    - for each issue create a merge request
    - clone the issue branch and work there
    - once the work is ready, submit your merge request via https://gitlab.ing.net/


- **Developing on master**
    - Easy work with, little git knowledge required
    - Higher chance of errors/bugs in `master` branch
    - More merge conflicts
    - Can't use merge requests for code review
    - In order to allow commits to master in ING's gitlab, you need to add `disable-peer-review` to settings > general > tags
- **Develop on dev, merge to master**
    - Requires basic knowledge of branching and merging
    - Errors/bugs can be caught before reaching `master`
    - Code reviews with merge requests are difficult, as everyone's work is in the same branch
- **Develop on feature branches, merge to master**
    - Requires disciplined team familiar with git
    - Errors/bugs can be caught and easily debugged within feature branch
    - Fewer merge conflicts with other features while developing
    - Code review is scoped to individual features/improvements, when using merge requests

    > We advise teams that are just starting out with git to use **Developing on master**, but later on they should keep the more advanced options in mind as they have some clear advantages.
    --->

###  Commit Messages

Before making commits, make sure that you have [https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup](configured your name and email)

When creating commit messages, keep these guidelines in mind. They are meant to keep the messages useful and relevant.

- Include a short description of *what* and *why* was done, *how* can be seen in the code.
- Use present tense, imperative mood
- Limit the length of the first line to 72 chars. You can use multiple messages to specify a second (longer) line: `git commit -m "Patch load function" -m "This is a much longer explanation of what was done"`
<!---- To refer to a Jira or Gitlab ticket, use **"See #123"** or **"Closes #123"**--->

To help you with this message format, you can have git generate new commit messages with template by running:

`git config commit.template .gitmessage`

Example message:
```
Create feature X
This is a longer description of what the commit contains. See #123
```

## Issue tracking

Issue tracking is recommended even for solo projects. In this project we use Gitlab issue board for issue tracking.

<!---
## (Optional) Testing: py.test
* To run a single testfile: `pytest tests/test_environment.py`
* Run all tests: `pytest`
* For informat: on on how to write tests using py.test, see the current examples and [https://docs.pytest.org/en/latest/](py.test's docs).
--->

## Versioning and Deployment

In the project we use versioning based on Semantic Versioning for Python. After a number of changes made to the repository, we make a deployment.

Deployment is made in the following steps:
- Merge a MR with the following changes
    - Bump the version in setup.py
    - Update Changelog with changes made from previous version
- Create a git tag for the version. This will trigger a CI pipeline which deploys  
