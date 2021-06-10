# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
    name='holidays-ext',
    version='0.0.4',
    description='Extended holidays package',
    url='https://github.com/kaixuyang/holidays-extension',
    author='Kaixu Yang',
    author_email='kaixuyang@gmail.com',
    license='MIT',
    packages=find_packages(),
    setup_requires=[
    ],
    install_requires=install_requires,
    python_requires='>=3',
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
