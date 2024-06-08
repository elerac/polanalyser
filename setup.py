from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="polanalyser",
    version="3.0.0",
    description="Polarization image analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elerac/polanalyser",
    author="Ryota Maeda",
    author_email="maeda.ryota.elerac@gmail.com",
    license="MIT",
    packages=find_packages(),
)
