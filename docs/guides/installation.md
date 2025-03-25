# Installation of the {{ toolkit }} Library

## Prerequisites

### Python

It is recommended that you use Python 3.10+. You can download and install the latest Python 
from the [python official website](https://www.python.org/downloads/).

### Package Manager

To install the `aigve` binaries, you will need to use pip. 

#### pip

If you installed Python via Homebrew or the Python website, pip (or pip3) was installed with it.

To install pip, you can refer to the [pip official website](https://pip.pypa.io/en/stable/installation/).

To upgrade your pip, you can use the following command:
```shell
python -m pip install --upgrade pip
```

--------------------

## Installation

The `aigve` library has been published at both [PyPI](https://pypi.org/project/aigve/) and the [project github repository](https://github.com/ShaneXiangH/AIGVE_Tool).

### Install from PyPI

To install `aigve` from [PyPI](https://pypi.org/project/aigve/), use the following command:

```shell
pip install aigve
```


### Install from Source Code

You can also install `aigve` from the source code, which has been released at the 
[project github repository](https://github.com/ShaneXiangH/AIGVE_Tool). 

You can download the public repository either from the project github webpage or via the following command:
```shell
git clone https://github.com/ShaneXiangH/AIGVE_Tool.git
```

After entering the downloaded source code directory, `aigve` can be installed with the following command:

```shell
python setup.py install
```

If you don't have `setuptools` installed locally, please consider to first install `setuptools`:
```shell
pip install setuptools 
```

### Dependency Packages

The `aigve` library is developed based on several dependency packages. 

**Option 1**: We recommend to create a specific conda environment for `AIGVE-Tool`:
```shell
conda env create -f environment.yml
conda activate aigve
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

--------------------

**Option 2**: The updated dependency [requirement.txt](https://github.com/ShaneXiangH/AIGVE_Tool/blob/main/requirement.txt) of `aigve`
can be downloaded from the [project github repository](https://github.com/ShaneXiangH/AIGVE_Tool).
After downloading the requirement.txt, you can install all these dependencies with the pip command:

=== "install command"
    ```shell
    pip install -r requirements.txt
    ```

=== "requirement.txt"
    ``` yaml linenums="1"
    mmcv==2.2.0
    mmdet==3.3.0
    mmengine==0.10.6
    transformers==4.46.3
    numpy==1.23.5
    pandas==2.2.3
    sympy==1.13.3
    einops==0.8.1
    sentencepiece==0.2.0
    h5py==3.12.1
    fvcore
    openai
    git+https://github.com/openai/CLIP.git
    tqdm
    pytorch==2.1.0 
    torchvision==0.16.0 
    torchaudio==2.1.0
    pytorchvideo
    gdown==5.2.0
    git+https://github.com/TIGER-AI-Lab/Mantis.git
    scipy==1.14.1
    decord==0.6.0
    ```


--------------------