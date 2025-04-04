# Tutorial on Running AIGV Evaluations

This tutorial explains how to run evaluations using the `AIGVE toolkit`. Once you have prepared your dataset, customized the dataloader, and implemented your metric module, you can configure and launch the AIGVE evaluation loop with ease.

## Prerequisites

Before running an evaluation, make sure:

* Your dataset is formatted and organized correctly ([Dataset Preparation](./dataset.md))

* Your dataloader is defined and included in a config file ([Customizable Dataloaders](./dataloader.md))

* Your metric is implemented and registered ([Modular Metrics](./evaluator.md))

* Your configuration is ready ([Configuration Files](./config.md))

## Running an Evaluation from a Configuration file

You can navigate to the AIGVE main directory and run your evaluation using the `main_aigve.py` script with your desired config:

```python
python main_aigve.py {metric_config_file}.py --work-dir {working_dir_path}
```

For example, to run the GSTVQA metric under its configuration file: 
```python
cd AIGVE_Tool/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/gstvqa.py --work-dir ./output
```

There are some other examples:

For SimpleVQA:
```python
cd AIGVE_Tool/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/simplevqa.py --work-dir ./output
```

For LightVQAPlus:
```python
cd AIGVE_Tool/aigve
python main_aigve.py AIGVE_Tool/aigve/configs/lightvqa_plus.py --work-dir ./output
```

**Note**: Some metrics require downloading pretrained models manually. Make sure they are downloaded correctly and placed in the correct paths as specified in the config (e.g., `model_path`).

Under our AIGVE Toolkit design, these running processes will:

* Load the dataset and dataloader

* Initialize the metric evaluator

* Run the evaluation loop

* Save the evaluation results

## Evaluation Output

The evaluation output depends on your metric class. Common output includes:

* Console logs (e.g., mean scores, per-sample results)

* Saved JSON file with results (e.g., `output_results.json`)

## Tips for Running AIGV Evaluations

* Start small: use toy version data and small configs to verify your setup.

* Check paths: ensure configuration file path and working directory path are correct.

* Some metrics require downloading pretrained models manually. Make sure they are downloaded correctly and placed in the correct paths as specified in the configuration files.

## What's Next?

Once your evaluation runs successfully, you can:

* Try running different existing metrics (you can choose from [here](https://github.com/ShaneXiangH/AIGVE_Tool/tree/main/aigve/configs))

* Benchmark on our AIGVE-Bench dataset (toy version is in [here](https://github.com/ShaneXiangH/AIGVE_Tool/tree/main/aigve/data/AIGVE_Bench_toy), the full version will publish soon!)

* Create your new AIGVE metrics following [Tutorial on customize evaluation metrics](./evaluator.md)

<!-- [Share your results on our leaderboard (coming soon!)] -->

Enjoy using AIGVE Toolkit for robust, modular, and reproducible AIGV evaluation!


