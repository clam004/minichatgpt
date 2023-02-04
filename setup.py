from setuptools import setup, find_packages

setup(
    name='minichatgpt',
    version='0.1.0',
    packages=find_packages(include=['minichatgpt', 'minichatgpt.*'])
)