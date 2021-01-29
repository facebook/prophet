## Copyright (c) 2017-present, Facebook, Inc.
## All rights reserved.

## This source code is licensed under the BSD-style license found in the
## LICENSE file in the root directory of this source tree. An additional grant
## of patent rights can be found in the PATENTS file in the same directory.


generated_holidays <- read.csv("data-raw/generated_holidays.csv")
usethis::use_data(generated_holidays, overwrite = TRUE)
