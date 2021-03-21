#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Robert Martin-Short",
    author_email="martinshortr@gmail.com",
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Build a generic fuzzy search engine for a corpus held in memory",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    keywords='genericsearch_test_empty',
    name="generic_search",
    packages=find_packages(include=["generic_search", "generic_search.*"]),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    include_package_data=True,
    package_data={"": ["tests/fixtures/IMDB Dataset,csv"]},
    version="0.1.0",
    zip_safe=False
)
