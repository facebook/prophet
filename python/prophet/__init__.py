# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from prophet.forecaster import Prophet

from pathlib import Path
about = {}
here = Path(__file__).parent.resolve()
with open(here / "__version__.py", "r") as f:
    exec(f.read(), about)
__version__ = about["__version__"]
