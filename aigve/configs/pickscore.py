# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.similarity_based import PickScore

with read_base():
    from ._base_.datasets.pickscore_dataset import *
    from ._base_.default import *


val_evaluator = dict(
    type=PickScore,
    model_name = "yuvalkirstain/PickScore_v1",
)

val_dataloader = dict(
    batch_size=4, 
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=PickScoreDataset,
        video_dir='AIGVE_Tool/data/toy/evaluate/',
        prompt_dir='AIGVE_Tool/data/toy/annotations/evaluate.json',
        processor_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
)
