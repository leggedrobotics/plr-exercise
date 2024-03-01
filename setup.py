from setuptools import find_packages
from distutils.core import setup

INSTALL_REQUIRES = [
    "numpy",
    "torch>=1.21",
]
setup(
    name="plr_exercise",
    version="1.0.0",
    author="Jonas Frey",
    author_email="jonfrey@ethz.ch",
    packages=find_packages(),
    python_requires=">=3.7",
    description="A small example package",
    install_requires=[INSTALL_REQUIRES],
)
