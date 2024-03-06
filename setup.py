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
    dependencies=[
        "https://download.pytorch.org/whl/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl"
    ],
)
