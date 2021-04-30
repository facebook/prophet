# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

Sys.setenv("R_TESTS" = "")
library(testthat)
library(prophet)

test_check("prophet")
