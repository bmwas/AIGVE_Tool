# Copyright (c) IFM Lab. All rights reserved.

from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.nn_based.simplevqa.simplevqa_metric import SimpleVQA
from datasets import SimpleVQADataset

with read_base():
    from ._base_.default import *

# Validation dataloader configuration
val_dataloader = dict(
    batch_size=1,  # One video per batch
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=SimpleVQADataset,
        video_dir='/home/xinhao/VQA_Toolkit/aigve/data/AIGVE_Bench/videos_3frame/',
        prompt_dir='/home/xinhao/VQA_Toolkit/aigve/data/AIGVE_Bench/annotations/test.json',
        max_len=8,  # Adjust based on video length
    )
)

# Evaluation configuration for SimpleVQA metric
val_evaluator = dict(
    type=SimpleVQA,
    is_gpu=True,
    model_spatial_path="metrics/video_quality_assessment/nn_based/simplevqa/UGC_BVQA_model.pth",
    model_motion_path="metrics/video_quality_assessment/nn_based/simplevqa/slowfast.pth",
)
