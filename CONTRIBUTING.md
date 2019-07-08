# Contributing guide


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
  * [Code for feature generation](#code-for-feature-generation)
  * [Functions for multiple modules](#functions-for-multiple-modules)
  
- [Feature Naming](#feature-naming)
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

- Notebooks are only allowed in the `notebooks/shared` and `notebooks/private` folder.
- The `notebooks/private` folder is git-ignored, hence there are no issues with your workflow. Use this directory for your development notebooks
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
- The `.py files` must be stored into the directory `src/mod_risk_fpd/` or a subdirectory of the same
- Written functions should follow the [Technical Standards](#technical-standards)




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
* Preferred method chaining style, especially relevant when using Spark:

```python
new_object = (
    old_object.first_method(args)
    .another_method()
    .and_another_one() # Maybe a little comment?
    .final_method(more_args)
)
```

### Code for feature generation
* Feature generation code is developed in Spark.</br>
* Every set of features that needs to be grouped together, needs to be in separate `.py` modules.
* Every python file used for feature generation, needs to be in `src/features`
* Every python module that calculates a set of features, needs to take in input the necessary `pyspark` dataframes.
Other necessary parameters are allowed, but they need to be predefined. 

* An example in pseudocode of a function generating the features is as follows:

```python 
def transactional_features(core_table, transactions):
    """
    Description of the function
    """
    # does operations with the tables and returns the feature
    
    return transactional__features
```
* Functions should not exceed 10-15 lines of code. If more code is needed, try to put together snippets of code into 
other functions. This make the code more readable, and easier to test.
* An example of the above point is shown in the pseudo code below (similar to the function description above) (assume
this code is stored in `src/features/example/py`)

```python 
import pyspark
from pyspark.sql import functions as F
# Do there all your imports

def transactional_features(core_table, transactions, minimum_transaction_amount = 100):
    """
    Pseudo function to illustrate the logic of the feature module.
    
    Args:
        core_table
        
        (pyspark.DataFrame): dataframe containing the debt episodes
        transactions: (pyspark.DataFrame):  dataframe containing the transactions of the customers
        minimum_transaction_amount (float): minimum amount of transactions to consider for building the features (default
        value is 100 TL)
    Returns:
        (pyspark.DataFrame): Features related to transactions.
    
    
    """
    # Here are some pseudo functions one might need 
    # Detect transactions related to the customers with the debt loans
     rel_transactions = get_relevant_transactions(core_table, transactions)
     
    # filter the transactions 
     rel_transactions = filter_transactions(rel_transactions, minimum_transaction_amount= minimum_transaction_amount)
     
    # aggregate transactions 
     transactional_features = aggregate_transactional_features(core_table, rel_transactions)
    
    
    return transactional_features
    
def get_relevant_transactions(core_table, transactions):
    """
    Pseudo function to illustrate the logic of the feature module. Example - retrieves only transactions that are needed
    
    Args:
        core_table (pyspark.DataFrame): dataframe containing the debt episodes
        transactions: (pyspark.DataFrame):  dataframe containing the transactions of the customers
        value is 100 TL)
    Returns:
        (pyspark.DataFrame): Dataframe with transactions that are present in the debt episodes
    
    
    """
    
    # no code provided there, just some pseudo logic -
    # you want to select only transactions of customers that have loans in arrears - probably by joining on the customer/products ids
    
    
def filter_transactions(transaction, minimum_transaction_amount):
    ### code of the funtcion
    
def aggregate_transactional_features(core_table, rel_transactions):
    ### code of the funtcion
    
    
```
In the Jupyter notebook, you would do something as follows:

First cell - relative inputs
```python
import sys
sys.path.insert(0,'../..')\

from src.features import example as ex
```

Next cell (calculate the )
```python
# calculate the transactional features
# note that only one function is imported
transact_feats = ex.transactional_features(core_table, transactions)

```
Next cell (add the features to the table)
```python
# Add the new features to your table

core_table = (
    core_table
    .join(transact_feats, how='left', on=['talep_id','inst_number']) 
    # left join is essential here - you might lose data otherwise
)

```

### Functions for multiple modules

Some functions might be useful for multiple features. Therefore, it does not make sense to store them in 
`feature_specific.py` file, but rather create add them in a helper file, i.e. `src/features/helpers.py`.<br>
An example:
- A function that helps with repartitioning the tables could be defined in `helpers.py`
- A function that helps with the naming convention (for example with the time interval definiton) could be
 defined in `helpers.py`
 - A function converts all IDs from String to Integers could be defined in `helpers.py`



### Feature Naming
#### General convention
It's important that feature naming follows a convention, so that every user can understand what the features means<br>
1. **feature names should be in English!** ;)
2. every feature name needs to start with `feat_`
3. the next set of string should indicate to which group of features does it apply. For example:
    - features related to the transactional account usage, should start with `feat_trx_` (created by `src/features/transactions.py`)
    - features related to the credit card used, should start with `feat_cc_` (created by `src/features/credit_card.py`)
    - features related to the previous debt episodes, should start with `feat_debt_` (created by `src/features/debt.py`)
    - features related to the previous payments, should start with `feat_pay_` (created by `src/features/.py`)
    
4. the following part of the name should describe the feature. When applicable, follow the following order:
    - Type of aggregations (`count`,`mean`,`std_dev`,`trend`...)
    - Filters (`inflows`,`leq_100_eur`)
    - Time windows (`D_-30_0`, `M_-2_0`) ([see below](#time-intervals-convention))
    - Any other information
    
    - As example: the features that counts the number of inflows in the range 100-200TL in 
    the 2 months previous to the debt episode should be named as follows:
    `feat_trx_count_inflow_heq_100_leq_200_M_-2_0` (which reads as  `features transactions counts inflows higher or 
    equal than 100 lower or equal than 200 in the months between 0 and 2)
    
    - Another example: average DPD in the 6 months previous to the debt episode should be named as follows:
    `feats_pay_mean_dpd_M_-6_0`
    
     - Another example: total DPD in the 6 months previous to the debt episode should be named as follows:
    `feats_pay_sum_dpd_M_-6_0`
    
    
  
#### Time intervals naming convention
When a time interval is used to compute a feature, it should follow the following conventions:
`<interval_type>_<start_interval_included_<end_interval_included>`

- <interval_type>: use `M` for calendar Months, `D` for Days
- <start_interval_included>: use integer positive number
- <end_interval_included>: use integer positive number

In every case discussed, 0 represents the day or calendar month of the debt episode. <br>
**Example**: <br>
if the debt episode is on the 5th of February 2018, then:
- `D 0` refers to the 5th of February 2018
- `D -1` refers to the 4th of February 2018
- `M 0` refers to the month of February 2018
- `M -2` refers fo December 2017

**Interval Example**:<br>
Keeping the previous example of the debt episode on the 5th of February 2018, if our features is computed on 
the following intervals:
- October 2017 (included) until January 2018 (included), then the feature should be named `M_-4_-1` (reads as
period from 4 calendar months ago to 1 calendar month ago, all included)
- 6th January 2018 (included) until 4th February 2018 (included), then the feature should be named `D_-30_-1` (reads as
period from 30 days ago to 1 day ago, all included)



#### Exceptions
 Of course, not all the features can follow the naming logic described above. Steps 1,2,3 of the General convention are
 still applicable, but step 4. might not be.<br>
 A few examples below:
    - requested amount: `feat_app_requested_amount` (Feature about the application requested amount)
    - postal code: `feat_soc_postal_code` (Feature about socio/demographic infromation postal_code)
    ...
    

    

### Line endings (optional)

Windows vs Linux line endings can be a huge source of trouble.
We have included a `.gitattributes` file that forces unix-style line endings (LF).

If you do run into trouble, please see our [dealing with windows line endings guide](https://confluence.europe.intranet/display/IAAT/Dealing+with+windows+line+endings+in+GIT).

## Project Data

- **Do not commit data to git!**. As git preserves history, onces data has been pushed to gitlab it is **very** difficult to remove.
- Very small .csv files can be permitted when used for testing, metadata or configuration purposes
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

There are a few common git workflows, depending on the project requirements and team maturity:

- **Development on user branches**:
This is not the standard procedure, but let's try to follow this method. If it does not work, we will change it.
    - Develop the code in your personal branch
    - Once the code is ready to be shared with the rest, create a new pull request
    - Ask one of your peers to review the code and merge the pull request
    
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
    --->

> We advise teams that are just starting out with git to use **Developing on master**, but later on they should keep the more advanced options in mind as they have some clear advantages.

###  Commit Messages

Before making commits, make sure that you have [https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup](configured your name and email)

When creating commit messages, keep these guidelines in mind. They are meant to keep the messages useful and relevant.

- Include a short description of *what* was done, *how* can be seen in the code.
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

<!---
## Issue tracking

** still under consideration **

Issue tracking is recommended even for solo projects. A single todo kanban boards can help structure and document your work. 

Currently we see being used:
- Gitlab issues and boards
- OrangeSharing JIRA tickets and boards
- Confluence JIRA tickets and boards

The best issue tracking system depends on your project. Gitlab is recommend if your project consists of mainly coders that have access to gitlab. 



## (Optional) Testing: py.test
* To run a single testfile: `pytest tests/test_environment.py`
* Run all tests: `pytest`
* For informat: on on how to write tests using py.test, see the current examples and [https://docs.pytest.org/en/latest/](py.test's docs).
--->