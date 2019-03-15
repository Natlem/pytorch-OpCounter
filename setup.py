#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages


readme = open('README.md').read()

VERSION = '0.0.22'

requirements = [
    'torch',
]

setup(
    # Metadata
    name='thop',
    version=VERSION,
    author='Le Thanh',
    author_email='nmlethanh91@gmail.com',
    url='https://github.com/Natlem/pytorch-OpCounter/',
    description='Forked of https://github.com/Lyken17/pytorch-OpCounter that can handle other architecture',
    long_description=readme,

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)