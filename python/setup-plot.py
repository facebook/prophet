import os.path
import pickle
import platform
import sys

from pkg_resources import (
    normalize_path,
    working_set,
    add_activation_listener,
    require,
)
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command


PLATFORM = 'unix'
if platform.platform().startswith('Win'):
    PLATFORM = 'win'

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SETUP_DIR, 'stan', PLATFORM)
MODELS_TARGET_DIR = os.path.join('fbprophet', 'stan_models')


def build_stan_models(target_dir, models_dir=MODELS_DIR):
    from pystan import StanModel
    for model_type in ['linear', 'logistic']:
        model_name = 'prophet_{}_growth.stan'.format(model_type)
        target_name = '{}_growth.pkl'.format(model_type)
        with open(os.path.join(models_dir, model_name)) as f:
            model_code = f.read()
        sm = StanModel(model_code=model_code)
        with open(os.path.join(target_dir, target_name), 'wb') as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)


class BuildPyCommand(build_py):
    """Custom build command to pre-compile Stan models."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODELS_TARGET_DIR)
            self.mkpath(target_dir)
            build_stan_models(target_dir)

        build_py.run(self)


class DevelopCommand(develop):
    """Custom develop command to pre-compile Stan models in-place."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.setup_path, MODELS_TARGET_DIR)
            self.mkpath(target_dir)
            build_stan_models(target_dir)

        develop.run(self)


version='0.1.1'
setup(
    name='fbprophet-plot',
    version=version,
    description='Automatic Forecasting Procedure',
    url='https://facebookincubator.github.io/prophet/',
    author='Sean J. Taylor <sjt@fb.com>, Ben Letham <bletham@fb.com>',
    author_email='sjt@fb.com',
    license='BSD',
    packages=['fbprophet-plot'],
    setup_requires=[
    ],
    install_requires=[
        'matplotlib',
        'pandas>=0.18.1',
        'pystan>=2.14',
        'fbprophet=={version}'.format(version=version),
    ],
    zip_safe=False,
    include_package_data=True,
    cmdclass={
        'build_py': BuildPyCommand,
        'develop': DevelopCommand,
    },
    long_description="""
Implements a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays.  It works best with daily periodicity data with at least one year of historical data.  Prophet is robust to missing data, shifts in the trend, and large outliers.
"""
)
