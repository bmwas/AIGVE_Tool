# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.similarity_based import BlipSimScore

with read_base():
    from ._base_.datasets.blipsim_dataset import *
    from ._base_.default import *


val_evaluator = dict(
    type=BlipSimScore,
    model_name="Salesforce/blip-itm-base-coco",
)

val_dataloader = dict(
    batch_size=4, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=BLIPSimDataset,
        processor_name='Salesforce/blip-itm-base-coco',
        video_dir='AIGVE_Tool/data/toy/evaluate/',
        prompt_dir='AIGVE_Tool/data/toy/annotations/evaluate.json',
    )
)
