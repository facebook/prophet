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
to improve, you need to learn how to work with GitHub and the *fbprophet* code base.


## Forking

You will need your own fork to work on the code. Go to the `fbprophet project
page <https://github.com/facebook/prophet>` and hit the ``Fork`` button. You will
want to clone your fork to your machine::

    git https://github.com/your-user-name/facebook/prophet.git
    cd prophet
    git remote add upstream https://github.com/facebook/prophet.git

This creates the directory `prophet` and connects your repository to
the upstream (main project) *prophet* repository.


## Creating a development environment

To test out code changes, you'll need to build *prophet* from source, which
requires a Python environment. 


### Creating a Python Environment

Before starting any development, you'll need to create an isolated prophet
development environment:

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_
- Make sure your conda is up to date (``conda update conda``)
- ``cd`` to the *prophet* source directory that you have cloned

1. Install the build dependencies

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

2. Build and install prophet

```
$ pip install -e .
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

See the full conda docs `here <http://conda.pydata.org/docs>`.

## Creating a branch

You want your master branch to reflect only production-ready code, so create a
feature branch for making your changes. For example:

```
$ git checkout -b shiny-new-feature
```

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *fbprophet*. You can have many "shiny-new-features"
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


<a id="documentation"> </a>

## Generating documentation

Most of the `doc` pages are generated from [Jupyter notebooks](http://jupyter.org/) in the [notebooks](https://github.com/facebook/prophet/tree/master/notebooks) directory at the base of the source tree.  Please make changes there and then rebuild the docs:

```
$ cd docs
$ make notebooks
```

Make sure you have installed [rpy2](https://rpy2.bitbucket.io/) so that the R code can be run as well.
