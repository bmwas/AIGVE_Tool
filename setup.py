# Copyright (c) IFM Lab. All rights reserved.

from setuptools import setup, find_packages

__version__ = '0.0.1'

requirements = [
    "torch==2.1.0",
    "torchvision==0.16.0",
    "torchaudio==0.12.1",
    "numpy==1.23.5",
    "scipy==1.14.1",
    "pandas==2.2.3",
    "pyyaml==6.0.2",
    "tqdm==4.67.1",
    "matplotlib==3.10.0",

    "mmcv==2.2.0",
    "mmdet==3.3.0",
    "mmengine==0.10.6",

    "transformers==4.46.3",
    "sentencepiece==0.2.0",
    "einops==0.8.1",
    "openai==1.65.1",
    "gdown==5.2.0",
    "fvcore==0.1.5.post20221221",
    "decord==0.6.0",
    "pytorchvideo==0.1.5",
    "clip==1.0",

    # Optional docs tools
    "mkdocs-material==9.6.5",
    "neoteroi-mkdocs==1.1.0",
    "mkdocs-macros-plugin==1.3.7",
    "mkdocs-jupyter==0.25.1",
    "mkdocstrings==0.28.2",
    "mkdocs-rss-plugin==1.17.1",
    "mkdocs-exclude==1.0.2",
    "mkdocstrings-python==1.16.2"
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aigve",  
    version=__version__,
    author="Xinhao Xiang, Xiao Liu, Zizhong Li, Zhuosheng Liu",
    author_email="xhxiang@ucdavis.edu",

    description="A comprehensive and structured evaluation framework for assessing AI-generated video quality.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://www.aigve.org",
    download_url="https://github.com/ShaneXiangH/AIGVE_Tool",
    packages=find_packages(include=["aigve", "aigve.*"]),

    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires='>=3.10',
    install_requires=requirements
)
