from enum import Enum
from typing import List


class StanBackendEnum(Enum):
    PYSTAN = 1
    CMDSTANPY = 2


def get_backends_from_env() -> List[StanBackendEnum]:
    import os
    backends = os.environ.get("STAN_BACKEND", StanBackendEnum.PYSTAN.name).split(",")
    res = []
    for x in backends:
        if x in StanBackendEnum.__members__:
            res.append(StanBackendEnum[x])
        else:
            raise RuntimeError("Stan backend not found: {0}".format(x))
    return res
