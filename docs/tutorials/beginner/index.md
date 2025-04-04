# Beginner's Tutorials


This series of beginner-friendly tutorials will guide you through using the AIGVE Toolkit to evaluate AI-generated videos in a modular, reproducible, and efficient manner.

You'll learn how to:

* Understand and work with Configuration Files

* Prepare your own datasets in AIGVE format 

* Customize dataloaders for your own datasets

* Implement or modify modular evaluation metrics 

* Run evaluations from both predefined and custom pipelines

Whether you're using AIGVE out-of-the-box or tailoring it for your own research, these tutorials will help you get started with confidence.

We assume you have correctly installed the latest `aigve` and its dependency packages already.
If you haven't installed them yet, please refer to the [installation](https://www.aigve.org/guides/installation/) page for the guidance.

This tutorial is prepared based on the 
[AIGVE-Tool paper](https://arxiv.org/abs/2503.14064). 
We also recommend reading that paper first for detailed technical information about `aigve` toolkit.


## Tutorial Organization

<!-- {% set quickstart_tutorial_files %}<a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/quickstart_tutorial.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a>  <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/notes/quickstart_tutorial.py"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a>{% endset %}
{% set expansion_tutorial_files %}<a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/expansion_tutorial.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/configs/expansion_function_postprocessing.yaml"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/expansion_tutorial.py"> <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"> </a>{% endset %}
{% set reconciliation_tutorial_files %}<a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/reconciliation_tutorial.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/configs/reconciliation_function_config.yaml"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/reconciliation_tutorial.py"> <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"> </a>{% endset %}
{% set data_interdependence_tutorial_files %}<a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/data_interdependence_tutorial.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/configs/data_interdependence_function_config.yaml"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/data_interdependence_tutorial.py"> <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"> </a>{% endset %}
{% set structural_interdependence_tutorial_files %}<a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/structural_interdependence_tutorial.ipynb"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/ipynb_icon.png" alt="Jupyter Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/configs/structural_interdependence_function_config.yaml"><img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/yaml_icon.png" alt="Yaml Logo" style="height: 2em; vertical-align: middle; margin-right: 4px;"></a> <a href="https://github.com/jwzhanggy/tinyBIG/blob/main/docs/tutorials/beginner/module/code/structural_interdependence_tutorial.py"> <img src="https://raw.githubusercontent.com/jwzhanggy/tinyBIG/main/docs/assets/img/python_icon.svg" alt="Python Logo" style="height: 2em; vertical-align: middle; margin-right: 10px;"> </a>{% endset %} -->

|                     Tutorial ID                      |            Tutorial Title            |   Released Date   |               
|:----------------------------------------------------:|:------------------------------------:|:-----------------:|
|    [Tutorial 0](../../guides/quick_start.md)       |         Quickstart Tutorial          |   March 29, 2025    |        
|     [Tutorial 1](./config.md)     |       Configuration Files       |   March 25, 2025    |        
|  [Tutorial 2](./dataset.md)   |  Dataset Preparation  | March 26, 2025 |     
|  [Tutorial 3](./dataloader.md)  |    Customizable Dataloaders    | March 27, 2025   | 
| [Tutorial 4](./evaluator.md) | Modular Metrics | March 29, 2025 | 
| [Tutorial 5](./running.md) | Running AIGV Evaluations  | March 29, 2025 | 