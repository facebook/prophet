Sys.setenv("R_TESTS" = "")
library(testthat)
library(prophet)

test_check("prophet")
