# Tutorial on Customizable Dataloaders


## Dataset Integration in Configuration Files

In the config file (e.g., `gstvqa.py`), the dataset is referenced as follows:

```python
val_dataloader = dict(
    dataset=dict(
        type=GSTVQADataset,
        video_dir='aigve/data/AIGVE_Bench/videos_3frame/',
        prompt_dir='aigve/data/AIGVE_Bench/annotations/test.json',
        ...
    )
)
```

Please make sure your `video_dir` contains the visual data, and `prompt_dir` points to your JSON annotations. For more information about customizing your dataloaders based on your prepared dataset, please refer to [Tutorial on Customizable Dataloaders](./dataloader.md).

---

This `GSTVQADataset` defined the unique logics about video data management, including handling diverse video sources reading, prompt retrieval, resolution normalization, frame sampling and padding, format conversion, feature extraction and pre-processing.