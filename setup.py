from setuptools import setup, find_packages

REQUIRED_PKGS = [
    "torch==2.5.0",
    "transformers==4.38.2",
    "numpy>=1.18.2",
    "accelerate",
    "datasets>=1.17.0",
    "tqdm",
]

setup(
    name='minichatgpt',
    version='0.1.0',
    packages=find_packages(include=['minichatgpt', 'minichatgpt.*']),
    install_requires=REQUIRED_PKGS,
    extras_require={
        'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    }
)