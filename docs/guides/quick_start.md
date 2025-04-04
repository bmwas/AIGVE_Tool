# Quickstart Tutorial

This quickstart tutorial walks you through running a ready-to-use evaluation in AIGVE Toolkit using existing configurations, datasets, and metrics. No customization is needed!

## Run Existing AIGVE Pipelines

AIGVE provides preconfigured examples so you can get started instantly. You can navigate to the `aigve/` folder and run:

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


## Check Your Outputs

After the run, check your working directory (e.g., `./output`) for:

* Log files with evaluation summaries

* Output JSON files containing per-video or aggregate metric results

## Tips for Quickstart

* Make sure your environment is installed correctly. Follow instructions in [installation](./installation.md).

* Check paths: ensure configuration file path and working directory path are correct.

* Some metrics require downloading pretrained models manually. Make sure they are downloaded correctly and placed in the correct paths as specified in the configuration files.