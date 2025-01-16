# VQA_Toolkit
a Video Quality Analysis Toolkit


## Implemented:

### Models:
#### NN based methods:
1. [GSTVQA](./configs/gstvqa.py) 
2. [ModularBVQA]()
3. [VideoPhy]()
4. [FID]()

#### Text-video alignment based methods:
1. [CLIPSim](./configs/clipsim.py) 

### Dataset:
1. [Toy dataset](./configs/_base_/datasets/toy_dataset.py) 


## Under progress:

### Models:
1. [StarVQA](./configs/starvqa.py)
2. [StarVQA_plus](./configs/starvqa.py)


### Dataset:
1. [KoNViD-1k](https://database.mmsp-kn.de/konvid-1k-database.html): 2.3GB
2. [Kinetics](https://github.com/cvdfoundation/kinetics-dataset): 63GB




## Implementing Suspended: 

### Models:
1. [StarVQA](./configs/starvqa.py): The pretrained model shared from author is out-of-date.

2. [StarVQA_plus](./configs/starvqa.py): need [Kinetics]() (63 GB for test split) and [LSVQ](https://github.com/baidut/PatchVQ) (need request from author) dataset. 

### Dataset:


## Environment

conda env remove --name vqa
```
conda env create -f environment.yml
conda activate vqa
```


## For developers:
### 0. Review the code of [VQALoop](./core/loops.py).

The loop generally consists of two parts: Dataloader and Evaluator. Inheritted from `torch.utils.data`, the Dataloader could load data in batch. Next, the Evaluator, which is inherited from the [BaseMetric](https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py#L16) class, could to the `process()` for each batch of the data and `evaluate()` the final scores in the final. We do not include the MODEL parts from MMEngine, since our Toolkit focus on the evaluation part of Video Quality Analysis , not the training part. All potential neural network `forword()` opertaions will be included in `process()` class of its Evaluator. 

### 1. Implement the Dataloader by inheritting the `torch.utils.data` class in the `\datasets` folder. 

#### 1.1 If the dataset is created by the origial author: directly use his/her Dataloader by putting it in the `\datasets` folder. (Example: [gstvqa_dataset](./datasets/gstvqa_dataset.py))

#### 1.2 If the dataset is a open-source dataset:

##### 1.2.1 Add the data in `\data` folder, and create the anootation .json file in [MM format](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). (Example: [toy dataset](.data/toy))

##### 1.2.2 Implement a Dataloader by inheritting the [MMEngine BaseDataset](https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py#L120) class (which then inherits the `torch.utils.data` class) in the `\datasets` folder. (Example: [toy_dataset](./datasets/gstvqa_dataset.py))

##### 1.2.3 Implement the dataset config files in `\config\_base_\datasets` folder. (Example: [toy_dataset](./configs/_base_/datasets/toy_dataset.py)) 

### 2. Implement the Evaluator in `\metrics` folder. 

#### 2.1 Identify its classcifation. (e.g. text_video_alignment or video_quality_assessment), create files in its directoy.

#### 2.2 Re-implement the metrics from the source by inheriting the [BaseMetric](https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py#L16) class. (Example: [GSTVQA](./configs/gstvqa.py), under its original source [code](https://github.com/Baoliang93/GSTVQA/blob/8463c9c3e5720349606d8efae7a5aa274bf69e7c/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py#L204))

#### 2.3 Implement the metric config files in `\config` folder. (Example: [gstvqa_dataset](./configs/gstvqa.py)) 

### 3. Finally, don't forget to add any new modules in its corresponding n`__init__.py` files.

### Useful operations:

``
git submodule add <github-repository-url> <folder-name>
``

## Run:
``
python main.py {metric_config_file}.py
``

Take Examples:

For GSTVQA:
``
cd VQA_Toolkit
python main.py /home/xinhao/VQA_Toolkit/configs/gstvqa.py --work-dir ./output
``

For CLIPSim:
``
python main.py /home/xinhao/VQA_Toolkit/configs/clipsim.py --work-dir ./output
``

For VideoPhy:
``
python main.py /home/xinhao/VQA_Toolkit/configs/clipsim.py --work-dir ./output
``

## To-do:

Frameworks written on Detectron2 frameworks:
1. LLM_Score 


## Acknowledge

The Toolkit is build top the top of [MMEngine](https://github.com/open-mmlab/mmengine)

We acknowledge original repositories of various VQA methods:
[GSTVQA](https://github.com/Baoliang93/GSTVQA),
[CLIPSim](https://github.com/zhengxu-1997/),
[ModularBVQA](https://github.com/winwinwenwen77/ModularBVQA),
<!-- [StarVQA](https://github.com/GZHU-DVL/StarVQA) -->