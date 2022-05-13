---
layout: docs
docid: "contributing"
title: "Getting Help and Contributing"
permalink: /docs/contributing.html
---

Prophet has a non-fixed release cycle but we will be making bugfixes in response to user feedback and adding features. Please let us know if you encounter a bug by [filing an issue](https://github.com/facebook/prophet/issues). Github issues is also the right place to ask questions about using Prophet.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features or extensions to the core, please first open an issue and discuss the feature with us. Sending a pull request is fine too, but it will likely be merged more quickly if any design decisions are settled on beforehand in an issue.

We try to keep the R and Python versions feature identical, but new features can be implemented for each method in separate commits.

The following sections will describe how you can submit a pull request for adding enhancements, documentation changes or bug fixes to the codebase.

## 1. Forking the Prophet Repo

You will need your own fork to work on the code. Go to the [prophet project
page](https://github.com/facebook/prophet) and hit the ``Fork`` button. You will
want to clone your fork to your machine:

```
$ git clone https://github.com/your-user-name/prophet.git
$ cd prophet
$ git remote add upstream https://github.com/facebook/prophet.git
```
This creates the directory `prophet` and connects your repository to
the upstream (main project) prophet repository.

## 2. Creating an environment with dependencies

Before starting any development, you'll need to create an isolated prophet
development environment. This should contain the required dependencies.

### Python

- Install either Anaconda [anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html)
- Make sure your conda is up to date (``conda update conda``)
- ``cd`` to the *prophet* source directory that you have cloned

```bash
$ cd python

# with Anaconda 
$ conda create -n prophet
$ conda activate prophet
$ pip install -r requirements.txt

# with venv
$ python3 -m venv prophet
$ source prophet/bin/activate
$ pip install -r requirements.txt
```

### R

Dependencies can be managed through [``Packrat``](https://rstudio.github.io/packrat/) or [``renv``](https://rstudio.github.io/renv/articles/renv.html).

For ``renv``, you must first initialise a new project local environment. 
```R
> setwd("path/to/prophet/R") # set R subdirectory as working directory
> install.packages('renv')
> renv::init()
```

This should also install the dependencies listed in the DESCRIPTION automatically. Any new R packages can be installed as they are needed in the project.

You can save the state of the project: 

```R
> renv::snapshot()
```

or load the environment: 

```R
> renv::restore()
```

## 3. Building a development version of Prophet

The next step is to build and install the development version of prophet in the environment you have just created. 

### Python

```bash
$ python setup.py develop
```

You should be able to import *prophet* from your locally built version:

```bash
$ python  # start an interpreter
>>> import prophet
>>> prophet.__version__
'1.0'  # whatever the current github version is
```

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation. 

```bash
# to view your environments:
$ conda info -e

# to return to your root environment::
$ conda deactivate
```

See the full conda docs [here](http://conda.pydata.org/docs).

### R

From the terminal, ``cd`` to ``R`` subdirectory and run:
```bash
$ R CMD INSTALL .
```

This will build and install the local version of the prophet package. Then from the R console you can load the package: ``library(prophet)``.

## 4. Creating a branch

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example:

```bash
$ git checkout -b new-feature
```

This changes your working directory to the new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *prophet*. You can have many "new-features"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the master branch:

```bash
$ git fetch upstream
$ git rebase upstream/master
```

This will replay your commits on top of the latest *prophet* git master.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``git stash`` them
prior to updating.  This will effectively store your changes and they can be
reapplied after updating.


## 5. Testing with Continuous Integration

Adding tests is one of the most common requests after code is pushed to prophet. Therefore, it is worth getting in the habit of writing tests ahead of time so this is never an issue. Once your pull request is submitted, the Github Actions CI (continuous integration) service automatically triggers Python and R builds for Prophet and runs the tests. A pull-request will be considered for merging when you have an all ‘green’ build. If any tests are failing, then you will get a red ‘X’, where you can click through to see the individual failed tests.

### Python

Prophet uses the ``unittest`` package for running tests in Python and ``testthat`` package for testing in R. All tests should go into the tests subdirectory in either the Python or R folders. 

The entire test suite can be run by typing: 
```bash
$ python setup.py tests
```

### R


The entire test suite can be run from the R console by installing ``devtools``:

```R
> install.packages('devtools')
> devtools::test()
```

Alternatively the test suite can be also run from the terminal after ``cd`` to the ``tests`` directory
```bash
$ Rscript testthat.R
```

or for just running a single test script like ``test_diagnostics.R`` from the R console:

```R
> library(testthat)
> source('test_diagnostics.R')
```

## 6. Generating documentation

Most of the `doc` pages are generated from [Jupyter notebooks](http://jupyter.org/) in the [notebooks](https://github.com/facebook/prophet/tree/master/notebooks) directory at the base of the source tree.  Please make changes there and then rebuild the docs:

```bash
$ cd docs
$ make notebooks
```

Make sure you have installed [rpy2](https://rpy2.bitbucket.io/) so that the R code can be run as well.

In R, the documentation for the source code must also generated if new parameters are added or a new function is created. This is documented with ``roxygen``. 

Run the command below before submitting a PR with any changes to the R code to update the function documentation:

```R
> devtools::document() 
```

## 7. Committing your code

Keep style fixes to a separate commit to make your pull request more readable. Once you’ve made changes, you can see them by typing:

```bash
$ git status
```

If you have created a new file, it is not being tracked by git. Add it by typing:

```bash
$ git add path/to/file-to-be-added.py
```

Doing ‘git status’ again should give something like:

```bash
# On branch new-feature
#
#       modified:   /relative/path/to/file-you-added.py
#
```

Now you can commit your changes in your local repository:

```bash
$ git commit -m
```

## 8. Pushing your changes

When you want your changes to appear publicly on your GitHub page, push your forked feature branch’s commits:

```bash
$ git push origin new-feature
```

Here origin is the default name given to your remote repository on GitHub. You can see the remote repositories:

```bash
$ git remote -v
```

If you added the upstream repository as described above you will see something like:

```bash
origin  git@github.com:yourname/prophet.git (fetch)
origin  git@github.com:yourname/prophet.git (push)
upstream	https://github.com/facebook/prophet.git (fetch)
upstream	https://github.com/facebook/prophet.git (push)
```

Now your code is on GitHub, but it is not yet a part of the prophet project. For that to happen, a pull request needs to be submitted on GitHub.

## 9. Review your code

When you’re ready to ask for a code review, file a pull request. Before you do, once again make sure that you have followed all the guidelines outlined in this document regarding code style, tests, performance tests, and documentation. You should also double check your branch changes against the branch it was based on:

1. Navigate to your repository on GitHub – https://github.com/your-user-name/prophet
2. Click on Branches
3. Click on the Compare button for your feature branch
4. Select the base and compare branches, if necessary. This will be master and new-feature, respectively.


## 10. Making a pull request

If everything looks good, you are ready to make a pull request. A pull request is how code from a local repository becomes available to the GitHub community and can be reviewed and eventually merged into the master version. This pull request and its associated changes will eventually be committed to the master branch and available in the next release. To submit a pull request:

1. Navigate to your repository on GitHub
2. Click on the Pull Request button
3. You can then click on Commits and Files Changed to make sure everything looks okay one last time
4. Write a description of your changes in the Preview Discussion tab
5. Click Send Pull Request.

This request then goes to the repository maintainers, and they will review the code. If you need to make more changes, you can make them in your branch, add them to a new commit, push them to GitHub, and the pull request will be automatically updated. Pushing them to GitHub again is done by:

```bash
$ git push origin new-feature
```

This will automatically update your pull request with the latest code and restart the Continuous Integration tests.


## 11. Delete your merged branch

Once your feature branch is merged into ``upstream master``, you can delete your remote branch via the ``Delete branch`` option in the PR and the local copy by running: 

```bash
$ git branch -d new-feature
```

## 12. PR checklist

* Write docstrings for any functions that are included.
* Test that the documentation builds correctly. See “Generating documentation”.
* Test your code.
  - Write new tests if needed. See "Testing with Continuous Integration"
  - Test the code using unittest. Running all tests takes a while, so feel free to only run the tests you think are needed based on your PR. CI will catch any failing tests.
* In R, you can also run ``devtools:check()`` for carrying out a number of automated checks all at once, for code problems, documentation, testing, package structure, vignettes etc. This will take a few minutes to run.
* Once you push your changes and make a PR, make sure you use an informative title which summarizes the changes you have made. 
* If the PR addresses an issue, please reference it e.g. fixes #1234
