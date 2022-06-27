import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EPyStateEstimator",
    version="0.1",
    author="Sanket Kamthe",
    description="Package for approximate inference in state estimation problems using Expectation Propagation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)