from setuptools import setup, find_packages


setup (

    packages=find_packages(),
    install_requires = open("MoeaBench/requirements.txt").read().splitlines(),


)