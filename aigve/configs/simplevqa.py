# Copyright (c) IFM Lab. All rights reserved.

from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.nn_based.simplevqa.simplevqa_metric import SimpleVQA
from datasets import SimpleVQADataset

with read_base():
    from ._base_.default import *

val_dataloader = dict(
    batch_size=1,  
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=SimpleVQADataset,
        # video_dir='AIGVE_Tool/aigve/data/toy/evaluate/', # it has 16 frames for each video, each frame is [512, 512, 3], it lasts about 2 seconds.
        # prompt_dir='AIGVE_Tool/aigve/data/toy/annotations/evaluate.json',
        video_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/videos/', # it has 81 frames for each video, each frame is [768, 1360, 3], it lasts about 5 seconds.
        prompt_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/annotations/train.json',
        min_video_seconds=8,  
    )
)

val_evaluator = dict(
    type=SimpleVQA,
    is_gpu=True,
    model_path="metrics/video_quality_assessment/nn_based/simplevqa/SimpleVQA/ckpts/UGC_BVQA_model.pth",
    # model_motion_path="metrics/video_quality_assessment/nn_based/simplevqa/slowfast.pth",
)
