# Contributing guide

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

* [Code development](#Code-development)
	* [Project workflow](#Project-workflow)
	* [Developing new functionality](#Developing-new-functionality)
	* [Ad-hoc development](#Ad-hoc-development)
	* [New contributors](#New-contributors)
* [Technical Standards](#Technical-Standards)
	* [Python](#Python)
	* [Code structure](#Code-structure)
	* [Functions in .py files](#Functions-in-.py-files)
	* [Probatus API](#Probatus-API)
	* [Unit tests](#Unit-tests)
	* [Jupyter notebooks](#Jupyter-notebooks)
	* [Line endings (optional)](#Line-endings-(optional))
	* [Documentation](#Documentation)
* [Project Data](#Project-Data)
* [Code Reviews](#Code-Reviews)
	* [What to review](#What-to-review)
* [Using GIT](#Using-GIT)
	* [Development workflow with GIT: Branching strategy](#Development-workflow-with-GIT:-Branching-strategy)
	* [ Commit Messages](#Commit-Messages)
* [Issue tracking](#Issue-tracking)
* [Versioning and Deployment](#Versioning-and-Deployment)

<!-- tocstop -->


## Code development

### Project workflow
For each new version of Probatus we loosely follow the steps:
* Set up a next release milestone, with the release date
* Plan issues that are picked up in a given milestone, and assign contributors
* Deploy new version of Probatus

Milestones are created by the Maintainer, in collaboration with contributors, based on contributors availability and priority of the backlog.

### Developing new functionality

For new functionality that can be developed in Probatus, we follow the workflow:
1. Make an issue on the board, clearly describe the idea and discuss it in the comments with other contributors.
2. Plan development of the feature, as part of one or multiple releases.
3. Implement the feature: feature code, unit tests and documentation (including notebook in `docs/` with tutorial on how to use the feature). Each of these can be implemented as a separate Merge Request.
4. Maintainer deploys new version of Probatus and expose it and new docs to the users.

### Ad-hoc development

For any other work picked up in the project, e.g. refactoring code, fixing bugs, refining docs we loosely follow this workflow:  
1. Make an issue on the board, clearly describe the idea and discuss it in the comments with other contributors.
2. Assign a contributor, and implement the change using Merge Request functionality.
3. Maintainer deploys new version of Probatus and expose it and new docs to the users.

### New contributors

We are always glad if new contributors want to spend time improving the package, do not hesitate to reach out to us.

For new contributors, a good starting point would be the following:
* Make an issues, describing and discussing ideas for improvements
* Look over issues that do not have an assignee, and assign yourself. Best to start with simple ones, e.g. improving docs, improving code style etc. This way you can get to know the package better.
* Create a Merge Request to master, that implements the issue that you are assigned to. Start with something small :)

## Technical Standards

### Python

* Use Python 3.6 or higher depends on availability in your dev/prod environment.
* Use environments (conda). To create one, run `conda env create -f requirements.yml`.
* Keep track of dependencies in `requirements.yml` and run `conda install --file requirements.yml` to install them. Remember to notify your colleagues of the change in requirements since they'll need to conda install them.
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

### Code structure
* Model validation modules assume that trained models passed for validation are developed in scikit-learn framework (have predict_proba and other standard functions), or follows scikit-learn API e.g. XGBoost.
* Every python file used for model validation, needs to be in `/probatus/`
* Class structure for a given module should have a base class, and specific functionality classes that inherit from base. If a given module implements only single way of computing the output, the base class is not required. 
* Functions should not exceed 10-15 lines of code. If more code is needed, try to put together snippets of code into 
other functions. This make the code more readable, and easier to test.

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
- Code is structured into folders, depending on the functionality. For code that is reusable across different modules, place it in `\utils\`.


### Probatus API
Classes follow the probatus API structure:
* Each class implements fit(), compute() and fit_compute() methods. Fit is used to fit object with provided data (unless no fit is required), and compute calculates the output e.g. DataFrame with report for the user. Lastly, fit_compute applies one after the other.
* If applicable, plot() method presents user with the appropriate graphs.
* For compute(), and plot(), check if the object is fitted first.
    
    
### Unit tests
Unit tests follow the following convention:
* Use pytest framework
* Structure tests in `tests/` folder the same way as `probatus/` modules. Each test file and function should start its name with `test_`.
* Do not store test data in git, use fixtures instead. For fixtures that can be reused across modules, store them in conftest.py, for others, keep them in the test file.
* **Make simple and quick unit tests. For most of them, it is best if you are able to compute the output by hand and make it deterministic.**
* For tests where functionality of other packages is run, or are undeterministic, consider mocking the functionality.
* Test most important units of code. If a single function calls multiple others, try to mock the function calls, and test those units separately.

### Jupyter notebooks

- Jupyter notebooks are used only to make documentation of the package.
- Do not collaborate on notebooks. Always create your own personal notebooks. This is because they are hard to version control as they include more than just code (cells & output).
- Do not copy and paste code. Instead write reusable functions in `src/` and load them into your notebooks (tip: use `%autoreload`). This blogpost details the workflow: [write less terrible notebook code](https://blog.godatadriven.com/write-less-terrible-notebook-code)


### Line endings (optional)

Windows vs Linux line endings can be a huge source of trouble.
We have included a `.gitattributes` file that forces unix-style line endings (LF).

If you do run into trouble, please see our [dealing with windows line endings guide](https://confluence.europe.intranet/display/IAAT/Dealing+with+windows+line+endings+in+GIT).

### Documentation

Documentation is a very crucial part of the project, because it ensures usability of the package.
We develop the docs in the following way:
* `docs/` folder contains all the relevant documentation
* We use sphinx-build to build html files (`site/` folder), based on `docs/`. If you want to build the `site/` folder, run the following in the root of the directory `sphinx-build -M html docs/source/ site/
`.
* For crucial functionality, we develop a jupyter notebooks, which serve as a tutorials for the users.
* Documentation is automatically deployed to `https://probatus.readthedocs.io/en/latest/`, whenever a new version of Probatus is created (using tag).

## Project Data

- **Do not commit data to git!**. As git preserves history, onces data has been pushed to gitlab it is **very** difficult to remove.
- Do not add pickle models into the repo, since they are dependent on the sklearn version used to initialize them
- Use the `/data` folder to store project data. This is .gitignore-ed directory by default
- If you need data for unit tests, initialize simple datasets as fixtures.

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

###  Commit Messages

Before making commits, make sure that you have [https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup](configured your name and email)

When creating commit messages, keep these guidelines in mind. They are meant to keep the messages useful and relevant.

- Include a short description of *what* and *why* was done, *how* can be seen in the code.
- Use present tense, imperative mood
- Limit the length of the first line to 72 chars. You can use multiple messages to specify a second (longer) line: `git commit -m "Patch load function" -m "This is a much longer explanation of what was done"`

To help you with this message format, you can have git generate new commit messages with template by running:

`git config commit.template .gitmessage`

Example message:
```
Create feature X
This is a longer description of what the commit contains. See #123
```

## Issue tracking

Issue tracking is recommended even for solo projects. In this project we use Gitlab issue board for issue tracking.


## Versioning and Deployment

In the project we use versioning based on Semantic Versioning for Python. After a number of changes made to the repository, we make a deployment.

Deployment is made in the following steps:
- Merge a MR with the following changes
    - Bump the version in setup.py
    - Bump release variable in docs/source/conf.py
    - Update Changelog with changes made from previous version
- Create a git tag for the version. This will trigger a CI pipeline which deploys 
- Deployment of new version is performed by Maintainers. 
