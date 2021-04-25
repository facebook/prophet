# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='fbprophet',
    version='1.0.1',
    description='Automatic Forecasting Procedure',
    url='https://facebook.github.io/prophet/',
    author='Sean J. Taylor <sjtz@pm.me>, Ben Letham <bletham@fb.com>',
    author_email='sjtz@pm.me',
    license='MIT',
    packages=find_packages(),
    setup_requires=[],
    install_requires=install_requires,
    python_requires='>=3',
    zip_safe=False,
    include_package_data=True,
    test_suite='fbprophet.tests',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
