import os.path
import pickle
import sys

from pkg_resources import (
    normalize_path,
    working_set,
    add_activation_listener,
    require,
)
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.test import test as test_command

class BuildPyCommand(build_py):
    """Custom build command to pre-compile Stan models."""

    def run(self):
        if not self.dry_run:
            self.build_stan_models()

        build_py.run(self)

    def build_stan_models(self):
        from pystan import StanModel
        target_dir = os.path.join(self.build_lib, 'fbprophet/stan_models')
        self.mkpath(target_dir)

        for model_type in ['linear', 'logistic']:
            with open('stan/prophet_{}_growth.stan'.format(model_type)) as f:
                model_code = f.read()
            sm = StanModel(model_code=model_code)
            with open(os.path.join(target_dir, '{}_growth.pkl'.format(model_type)), 'wb') as f:
                pickle.dump(sm, f)

class TestCommand(test_command):
    """We must run tests on the build directory, not source."""

    def with_project_on_sys_path(self, func):
        # Ensure metadata is up-to-date
        self.reinitialize_command('build_py', inplace=0)
        self.run_command('build_py')
        bpy_cmd = self.get_finalized_command("build_py")
        build_path = normalize_path(bpy_cmd.build_lib)

        # Build extensions
        self.reinitialize_command('egg_info', egg_base=build_path)
        self.run_command('egg_info')

        self.reinitialize_command('build_ext', inplace=0)
        self.run_command('build_ext')

        ei_cmd = self.get_finalized_command("egg_info")

        old_path = sys.path[:]
        old_modules = sys.modules.copy()

        try:
            sys.path.insert(0, normalize_path(ei_cmd.egg_base))
            working_set.__init__()
            add_activation_listener(lambda dist: dist.activate())
            require('%s==%s' % (ei_cmd.egg_name, ei_cmd.egg_version))
            func()
        finally:
            sys.path[:] = old_path
            sys.modules.clear()
            sys.modules.update(old_modules)
            working_set.__init__()

setup(
    name='fbprophet',
    version='0.1.post1',
    description='Automatic Forecasting Procedure',
    url='https://facebookincubator.github.io/prophet/',
    author='Sean J. Taylor <sjt@fb.com>, Ben Letham <bletham@fb.com>',
    author_email='sjt@fb.com',
    license='BSD',
    packages=['fbprophet', 'fbprophet.tests'],
    setup_requires=[
        'Cython>=0.22',
        'pystan>=2.8',
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'pystan>=2.8',
    ],
    zip_safe=False,
    include_package_data=True,
    cmdclass={
        'build_py': BuildPyCommand,
        'test': TestCommand,
    },
    test_suite='fbprophet.tests.test_prophet',
    long_description="""
Implements a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays.  It works best with daily periodicity data with at least one year of historical data.  Prophet is robust to missing data, shifts in the trend, and large outliers.
"""
)
