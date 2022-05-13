# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import platform
import sys
import os
from pkg_resources import (
    normalize_path,
    working_set,
    add_activation_listener,
    require,
)
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.test import test as test_command
from typing import List

PLATFORM = 'unix'
if platform.platform().startswith('Win'):
    PLATFORM = 'win'

MODEL_DIR = os.path.join('stan', PLATFORM)
MODEL_TARGET_DIR = os.path.join('prophet', 'stan_model')


def get_backends_from_env() -> List[str]:
    from prophet.models import StanBackendEnum
    return os.environ.get("STAN_BACKEND", StanBackendEnum.PYSTAN.name).split(",")


def build_models(target_dir):
    from prophet.models import StanBackendEnum
    for backend in get_backends_from_env():
        StanBackendEnum.get_backend_class(backend).build_model(target_dir, MODEL_DIR)


class BuildPyCommand(build_py):
    """Custom build command to pre-compile Stan models."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            build_models(target_dir)

        build_py.run(self)


class DevelopCommand(develop):
    """Custom develop command to pre-compile Stan models in-place."""

    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.setup_path, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            build_models(target_dir)

        develop.run(self)


class TestCommand(test_command):
    user_options = [
        ('test-module=', 'm', "Run 'test_suite' in specified module"),
        ('test-suite=', 's',
         "Run single test, case or suite (e.g. 'module.test_suite')"),
        ('test-runner=', 'r', "Test runner to use"),
        ('test-slow', 'w', "Test slow suites (default off)"),
    ]
    test_slow = None

    def initialize_options(self):
        super(TestCommand, self).initialize_options()
        self.test_slow = False

    def finalize_options(self):
        super(TestCommand, self).finalize_options()
        if self.test_slow is None:
            self.test_slow = getattr(self.distribution, 'test_slow', False)

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

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='prophet',
    version='1.0.1',
    description='Automatic Forecasting Procedure',
    url='https://facebook.github.io/prophet/',
    author='Sean J. Taylor <sjtz@pm.me>, Ben Letham <bletham@fb.com>',
    author_email='sjtz@pm.me',
    license='MIT',
    packages=find_packages(),
    setup_requires=[
    ],
    install_requires=install_requires,
    python_requires='>=3',
    zip_safe=False,
    include_package_data=True,
    cmdclass={
        'build_py': BuildPyCommand,
        'develop': DevelopCommand,
        'test': TestCommand,
    },
    test_suite='prophet.tests',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
