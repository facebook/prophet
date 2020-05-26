---
layout: docs
docid: "contributing"
title: "Getting Help and Contributing"
permalink: /docs/contributing.html
subsections:
  - id: documentation
    title: Generating documentation
---

Prophet has a non-fixed release cycle but we will be making bugfixes in response to user feedback and adding features.  Its current state is Beta (v0.5), we expect no obvious bugs. Please let us know if you encounter a bug by [filing an issue](https://github.com/facebook/prophet/issues). Github issues is also the right place to ask questions about using Prophet.

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features or extensions to the core, please first open an issue and discuss the feature with us. Sending a pull request is fine too, but it will likely be merged more quickly if any design decisions are settled on beforehand in an issue.

The R and Python versions are kept feature identical, but new features can be implemented for each method in separate commits.


## Making a pull request 

Now that you have an issue you want to fix, enhancement to add, or documentation
to improve, you need to learn how to work with GitHub and the fbprophet code base.


## Forking

You will need your own fork to work on the code. Go to the fbprophet project
page https://github.com/facebook/prophet and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git https://github.com/your-user-name/prophet.git
    cd prophet
    git remote add upstream https://github.com/facebook/prophet.git

This creates the directory `prophet` and connects your repository to
the upstream (main project) fbprophet repository.


## Creating a development environment

Before starting any development, you'll need to create an isolated prophet
development environment. This should contain the required dependencies and 
the development version of fbprophet

### Installing a new environment with dependencies.

#### Python

- Install either Anaconda (https://www.anaconda.com/download/) or miniconda (https://conda.io/miniconda.html)
- Make sure your conda is up to date (``conda update conda``)
- ``cd`` to the *prophet* source directory that you have cloned

```
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

#### R

Dependencies can be managed through ``Packrat`` (https://rstudio.github.io/packrat/) or ``renv`` (https://rstudio.github.io/renv/articles/renv.html).

For ``renv`` , you must first initialise a new project local environment. 
```
renv::init()
```

This should also install the dependencies listed in the NAMESPACE automatically. Any new R packages can be installed as they are needed in the project e.g. ``install.pacakges('devtools')``

You can save the state of the project: 

```
renv::snapshot()
```

or load the environment: 

```
renv::restore()
```


### Build and install the development version of prophet

#### Python

```
$ python setup.py develop
```

You should be able to import *fbprophet* from your locally built version:

```
$ python  # start an interpreter
>>> import fbprophet
>>> fbprophet.__version__
'0.6.1.dev0
'0.10.0+dev46.g015daca'
```

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation. 

```
# to view your environments:
$ conda info -e

# to return to your root environment::
$ conda deactivate
```

See the full conda docs here http://conda.pydata.org/docs.

#### R

```
R CMD BUILD 
R CMD INSTALL
```

Then start R and type library(prophet) to see that it was indeed installed, and then try out one of the functions. 

## Creating a branch

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example:

```
$ git checkout -b new-feature
```

This changes your working directory to the new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *fbprophet*. You can have many "new-features"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the master branch:

```
$ git fetch upstream
$ git rebase upstream/master
```

This will replay your commits on top of the latest *fbprophet* git master.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``git stash`` them
prior to updating.  This will effectively store your changes and they can be
reapplied after updating.


## Testing with Continuous Integration

fbprophet is serious about testing and strongly encourages contributors to embrace test-driven development (TDD). This development process “relies on the repetition of a very short development cycle: first the developer writes an (initially failing) automated test case that defines a desired improvement or new function, then produces the minimum amount of code to pass that test.” So, before actually writing any code, you should write your tests. Often the test can be taken from the original GitHub issue. However, it is always worth considering additional use cases and writing corresponding tests.

Adding tests is one of the most common requests after code is pushed to xarray. Therefore, it is worth getting in the habit of writing tests ahead of time so this is never an issue. The prophet test suite runs automatically the Azure Pipelines, continuous integration service, once your pull request is submitted. A pull-request will be considered for merging when you have an all ‘green’ build. If any tests are failing, then you will get a red ‘X’, where you can click through to see the individual failed tests.

#### Python

Prophet uses the ``unnittest`` package for running tests in Python and ``testthat`` package for testing in R. All tests should go into the tests subdirectory in either the Python or R folders. 


The entire test suite can be run by typing: 
```
python setup.py tests
```

#### R


The entire test suite can be run from R or the terminal by typing:

```
devtools::test()

# or if running from the terminal after ``cd`` to the ``tests`` directory

$ Rscript testthat.R
```

or just running a single test script:

```

Rscript testthat.R
```

## Committing your code

Keep style fixes to a separate commit to make your pull request more readable. Once you’ve made changes, you can see them by typing:

```
git status
```

If you have created a new file, it is not being tracked by git. Add it by typing:

```
git add path/to/file-to-be-added.py
```

Doing ‘git status’ again should give something like:

```
# On branch new-feature
#
#       modified:   /relative/path/to/file-you-added.py
#
```

Now you can commit your changes in your local repository:

```
git commit -m
```

## Pushing your changes

When you want your changes to appear publicly on your GitHub page, push your forked feature branch’s commits:

```
git push origin new-feature
```

Here origin is the default name given to your remote repository on GitHub. You can see the remote repositories:

```
git remote -v
```

If you added the upstream repository as described above you will see something like:

```
origin  git@github.com:yourname/prophet.git (fetch)
origin  git@github.com:yourname/prophet.git (push)
upstream	https://github.com/facebook/prophet.git (fetch)
upstream	https://github.com/facebook/prophet.git (push)
```

Now your code is on GitHub, but it is not yet a part of the fbprophet project. For that to happen, a pull request needs to be submitted on GitHub.

## Review your code

When you’re ready to ask for a code review, file a pull request. Before you do, once again make sure that you have followed all the guidelines outlined in this document regarding code style, tests, performance tests, and documentation. You should also double check your branch changes against the branch it was based on:

1. Navigate to your repository on GitHub – https://github.com/your-user-name/prophet
2. Click on Branches
3. Click on the Compare button for your feature branch
4. Select the base and compare branches, if necessary. This will be master and new-feature, respectively.


## Finally, make the pull request

If everything looks good, you are ready to make a pull request. A pull request is how code from a local repository becomes available to the GitHub community and can be looked at and eventually merged into the master version. This pull request and its associated changes will eventually be committed to the master branch and available in the next release. To submit a pull request:

1. Navigate to your repository on GitHub

2. Click on the Pull Request button

3. You can then click on Commits and Files Changed to make sure everything looks okay one last time

4. Write a description of your changes in the Preview Discussion tab

5. Click Send Pull Request.

This request then goes to the repository maintainers, and they will review the code. If you need to make more changes, you can make them in your branch, add them to a new commit, push them to GitHub, and the pull request will be automatically updated. Pushing them to GitHub again is done by:

```
git push origin new-feature
```

This will automatically update your pull request with the latest code and restart the Continuous Integration tests.


## Delete your merged branch (optional)

Once your feature branch is accepted into upstream, you’ll probably want to get rid of the branch. First, merge upstream master into your branch so git knows it is safe to delete your branch:

```
git fetch upstream
git checkout master
git merge upstream/master
```

Then you can do:

```
git branch -d new-feature
```

Make sure you use a lower-case -d, or else git won’t warn you if your feature branch has not actually been merged.


<a id="documentation"> </a>

## Generating documentation

Most of the `doc` pages are generated from [Jupyter notebooks](http://jupyter.org/) in the [notebooks](https://github.com/facebook/prophet/tree/master/notebooks) directory at the base of the source tree.  Please make changes there and then rebuild the docs:

```
$ cd docs
$ make notebooks
```

Make sure you have installed [rpy2](https://rpy2.bitbucket.io/) so that the R code can be run as well.

In R, the documentation for the source code must also generated if new parameters are added or a new function is created. This is documented with ``roxygen``. 

Run the command below before submitting a PR with any changes to the R code, otherwise the CI check will error:

```
devtools::document() 
```

## PR checklist

* Write docstrings for any functions that are included.

* Test that the documentation builds correctly. See “Generating documentation”.

* Test your code.
  - Write new tests if needed. See "Testing with Continuous Integration"
  - Test the code using unittest. Running all tests takes a while, so feel free to only run the tests you think are needed based on your PR. CI will catch any failing tests.

* Push your code and create a PR on GitHub.
* Use a helpful title for your pull request by summarizing the main contributions rather than using the latest commit message. If this addresses an issue, please reference it.
