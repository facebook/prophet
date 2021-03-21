# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from prophet.forecaster import Prophet

logger = logging.getLogger('fbprophet')

logger.warning(
    'As of v1.0, the package name has changed from "fbprophet" to "prophet". '
    'Please update references in your code accordingly.'
)
