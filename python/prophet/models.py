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
import pickle
import pkg_resources
import os

import logging
logger = logging.getLogger('prophet.models')


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

    @staticmethod
    @abstractmethod
    def build_model(target_dir, model_dir):
        pass


class CmdStanPyBackend(IStanBackend):

    @staticmethod
    def get_type():
        return StanBackendEnum.CMDSTANPY.name

    @staticmethod
    def build_model(target_dir, model_dir):
        from shutil import copy
        import cmdstanpy
        model_name = 'prophet.stan'
        target_name = 'prophet_model.bin'

        sm = cmdstanpy.CmdStanModel(
            stan_file=os.path.join(model_dir, model_name))
        sm.compile()
        copy(sm.exe_file, os.path.join(target_dir, target_name))

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
            if self.newton_fallback and args['algorithm'] != 'Newton':
                logger.warning(
                    'Optimization terminated abnormally. Falling back to Newton.'
                )
                args['algorithm'] = 'Newton'
                self.stan_fit = self.model.optimize(**args)
            else:
                raise e

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
            algorithm='Newton' if stan_data['T'] < 100 else 'LBFGS',
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
            if "." in cname:
                parsed = cname.split(".")
            else:
                parsed = cname.split("[")

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
            else:
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


class PyStanBackend(IStanBackend):

    @staticmethod
    def get_type():
        return StanBackendEnum.PYSTAN.name

    @staticmethod
    def build_model(target_dir, model_dir):
        import pystan
        model_name = 'prophet.stan'
        target_name = 'prophet_model.pkl'
        with open(os.path.join(model_dir, model_name)) as f:
            model_code = f.read()
        sm = pystan.StanModel(model_code=model_code)
        with open(os.path.join(target_dir, target_name), 'wb') as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)

    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:

        args = dict(
            data=stan_data,
            init=lambda: stan_init,
            iter=samples,
        )
        args.update(kwargs)
        self.stan_fit = self.model.sampling(**args)
        out = {}
        for par in self.stan_fit.model_pars:
            out[par] = self.stan_fit[par]
            # Shape vector parameters
            if par in ['delta', 'beta'] and len(out[par].shape) < 2:
                out[par] = out[par].reshape((-1, 1))
        return out

    def fit(self, stan_init, stan_data, **kwargs) -> dict:

        args = dict(
            data=stan_data,
            init=lambda: stan_init,
            algorithm='Newton' if stan_data['T'] < 100 else 'LBFGS',
            iter=1e4,
        )
        args.update(kwargs)
        try:
            self.stan_fit = self.model.optimizing(**args)
        except RuntimeError as e:
            # Fall back on Newton
            if self.newton_fallback and args['algorithm'] != 'Newton':
                logger.warning(
                    'Optimization terminated abnormally. Falling back to Newton.'
                )
                args['algorithm'] = 'Newton'
                self.stan_fit = self.model.optimizing(**args)
            else:
                raise e

        params = {}

        for par in self.stan_fit.keys():
            params[par] = self.stan_fit[par].reshape((1, -1))

        return params

    def load_model(self):
        """Load compiled Stan model"""
        model_file = pkg_resources.resource_filename(
            'prophet',
            'stan_model/prophet_model.pkl',
        )
        with Path(model_file).open('rb') as f:
            return pickle.load(f)


class StanBackendEnum(Enum):
    PYSTAN = PyStanBackend
    CMDSTANPY = CmdStanPyBackend

    @staticmethod
    def get_backend_class(name: str) -> IStanBackend:
        try:
            return StanBackendEnum[name].value
        except KeyError as e:
            raise ValueError("Unknown stan backend: {}".format(name)) from e
