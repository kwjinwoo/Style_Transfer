from setuptools import find_packages, setup

setup(
    name="style_transfer",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
