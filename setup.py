from setuptools import setup, find_packages

setup(
    author="Robert Martin-Short",
    author_email="martinshortr@gmail.com",
    description="Build a generic fuzzy search engine for a corpus held in memory",
    name="generic_search",
    version="0.1.0",
    packages=find_packages(include=["generic_search", "generic_search.*"]),
    include_package_data=True,
    package_data={"": ["tests/fixtures/IMDB Dataset,csv"]}
)
