---
layout: docs
docid: "contributing"
title: "How to Contribute"
permalink: /docs/contributing.html
---

Prophet has an non-fixed release cycle but we will be making bugfixes in response to user feedback and adding features.  Its current state is Beta (v0.2), we expect no obvious bugs. Please let us know if you encounter a bug by [filing an issue](https://github.com/facebook/prophet/issues).

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features or extensions to the core, please first open an issue and discuss the feature with us. Sending a pull request is fine, too, but if it is a large change we suggest you run it by us first.

We require that any API changes or feature additions are made available for both Python and R in parallel.

## Documentation

Most of the `doc` pages are generated from [Jupyter notebooks](http://jupyter.org/) in the [notebooks](https://github.com/facebook/prophet/tree/master/notebooks) directory at the base of the source tree.  Please make changes there and then rebuild the docs:

```
$ cd docs
$ make notebooks
```

Make sure you have installed [rpy2](https://rpy2.bitbucket.io/) so that the R code can be run as well.
