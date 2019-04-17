import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mykit",
    version=__version__,
    author="sparsh agarwal",
    author_email="sp.agarwal13@gmail.com",
    description="my useful function package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sparsh9012/mykit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
