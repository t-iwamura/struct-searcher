from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="struct_searcher",
    version="1.1.0",
    author="Taiki Iwamura",
    author_email="takki.0206@gmail.com",
    description="Python package for structural search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/t-iwamura/struct-searcher",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "click",
        "lammps",
        "pymatgen",
    ],
    entry_points={
        "console_scripts": [
            "struct-searcher=struct_searcher.scripts.main:main",
        ],
    },
)
