from setuptools import setup, find_packages

setup(
    author="Robert Martin-Short",
    description="Build a generic fuzzy search engine for a corpus held in memory",
    name="generic_search",
    version="0.1.0",
    packages=find_packages(include=["generic_search","generic_search.*"])
)