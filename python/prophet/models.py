# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from abc import abstractmethod, ABC
from typing import Tuple
from collections import OrderedDict
from enum import Enum
from pathlib import Path
import pkg_resources
import platform

import logging
logger = logging.getLogger('prophet.models')

PLATFORM = "win" if platform.platform().startswith("Win") else "unix"

class IStanBackend(ABC):
    def __init__(self):
        self.model = self.load_model()
        self.stan_fit = None
        self.newton_fallback = True

    def set_options(self, **kwargs):
        """
        Specify model options as kwargs.
         * newton_fallback [bool]: whether to fallback to Newton if L-BFGS fails
        """
        for k, v in kwargs.items():
            if k == 'newton_fallback':
                self.newton_fallback = v
            else:
                raise ValueError(f'Unknown option {k}')


    @staticmethod
    @abstractmethod
    def get_type():
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def fit(self, stan_init, stan_data, **kwargs) -> dict:
        pass

    @abstractmethod
    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        pass


class CmdStanPyBackend(IStanBackend):
    CMDSTAN_VERSION = "2.26.1"
    def __init__(self):
        import cmdstanpy
        # this must be set before super.__init__() for load_model to work on Windows
        local_cmdstan = pkg_resources.resource_filename(
            "prophet", f"stan_model/cmdstan-{self.CMDSTAN_VERSION}"
        )
        if Path(local_cmdstan).exists():
            cmdstanpy.set_cmdstan_path(local_cmdstan)
        super().__init__()

    @staticmethod
    def get_type():
        return StanBackendEnum.CMDSTANPY.name

    def load_model(self):
        import cmdstanpy
        model_file = pkg_resources.resource_filename(
            'prophet',
            'stan_model/prophet_model.bin',
        )
        return cmdstanpy.CmdStanModel(exe_file=model_file)

    def fit(self, stan_init, stan_data, **kwargs):
        (stan_init, stan_data) = self.prepare_data(stan_init, stan_data)

        if 'inits' not in kwargs and 'init' in kwargs:
            kwargs['inits'] = self.prepare_data(kwargs['init'], stan_data)[0]

        args = dict(
            data=stan_data,
            inits=stan_init,
            algorithm='Newton' if stan_data['T'] < 100 else 'LBFGS',
            iter=int(1e4),
        )
        args.update(kwargs)

        try:
            self.stan_fit = self.model.optimize(**args)
        except RuntimeError as e:
            # Fall back on Newton
            if not self.newton_fallback or args['algorithm'] == 'Newton':
                raise e
            logger.warning('Optimization terminated abnormally. Falling back to Newton.')
            args['algorithm'] = 'Newton'
            self.stan_fit = self.model.optimize(**args)
        params = self.stan_to_dict_numpy(
            self.stan_fit.column_names, self.stan_fit.optimized_params_np)
        for par in params:
            params[par] = params[par].reshape((1, -1))
        return params

    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        (stan_init, stan_data) = self.prepare_data(stan_init, stan_data)

        if 'inits' not in kwargs and 'init' in kwargs:
            kwargs['inits'] = self.prepare_data(kwargs['init'], stan_data)[0]

        args = dict(
            data=stan_data,
            inits=stan_init,
        )

        if 'chains' not in kwargs:
            kwargs['chains'] = 4
        iter_half = samples // 2
        kwargs['iter_sampling'] = iter_half
        if 'iter_warmup' not in kwargs:
            kwargs['iter_warmup'] = iter_half

        args.update(kwargs)

        self.stan_fit = self.model.sample(**args)
        res = self.stan_fit.draws()
        (samples, c, columns) = res.shape
        res = res.reshape((samples * c, columns))
        params = self.stan_to_dict_numpy(self.stan_fit.column_names, res)

        for par in params:
            s = params[par].shape
            if s[1] == 1:
                params[par] = params[par].reshape((s[0],))

            if par in ['delta', 'beta'] and len(s) < 2:
                params[par] = params[par].reshape((-1, 1))

        return params

    @staticmethod
    def prepare_data(init, data) -> Tuple[dict, dict]:
        cmdstanpy_data = {
            'T': data['T'],
            'S': data['S'],
            'K': data['K'],
            'tau': data['tau'],
            'trend_indicator': data['trend_indicator'],
            'y': data['y'].tolist(),
            't': data['t'].tolist(),
            'cap': data['cap'].tolist(),
            't_change': data['t_change'].tolist(),
            's_a': data['s_a'].tolist(),
            's_m': data['s_m'].tolist(),
            'X': data['X'].to_numpy().tolist(),
            'sigmas': data['sigmas']
        }

        cmdstanpy_init = {
            'k': init['k'],
            'm': init['m'],
            'delta': init['delta'].tolist(),
            'beta': init['beta'].tolist(),
            'sigma_obs': init['sigma_obs']
        }
        return (cmdstanpy_init, cmdstanpy_data)

    @staticmethod
    def stan_to_dict_numpy(column_names: Tuple[str, ...], data: 'np.array'):
        import numpy as np

        output = OrderedDict()

        prev = None

        start = 0
        end = 0
        two_dims = len(data.shape) > 1
        for cname in column_names:
            parsed = cname.split(".") if "." in cname else cname.split("[")
            curr = parsed[0]
            if prev is None:
                prev = curr

            if curr != prev:
                if prev in output:
                    raise RuntimeError(
                        "Found repeated column name"
                    )
                if two_dims:
                    output[prev] = np.array(data[:, start:end])
                else:
                    output[prev] = np.array(data[start:end])
                prev = curr
                start = end
            end += 1
        if prev in output:
            raise RuntimeError(
                "Found repeated column name"
            )
        if two_dims:
            output[prev] = np.array(data[:, start:end])
        else:
            output[prev] = np.array(data[start:end])
        return output




class StanBackendEnum(Enum):
    CMDSTANPY = CmdStanPyBackend

    @staticmethod
    def get_backend_class(name: str) -> IStanBackend:
        try:
            return StanBackendEnum[name].value
        except KeyError as e:
            raise ValueError(f"Unknown stan backend: {name}") from e
