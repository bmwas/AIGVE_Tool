# Copyright (c) IFM Lab. All rights reserved.

from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.nn_based.lightvqa_plus.lightvqa_plus_metric import LightVQAPlus
from datasets import LightVQAPlusDataset

with read_base():
    from ._base_.default import *

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=LightVQAPlusDataset,
        # video_dir='AIGVE_Tool/aigve/data/toy/evaluate/', # it has 16 frames for each video, each frame is [512, 512, 3], it lasts about 2 seconds.
        # prompt_dir='AIGVE_Tool/aigve/data/toy/annotations/evaluate.json',
        video_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/videos/', # it has 81 frames for each video, each frame is [768, 1360, 3], it lasts about 5 seconds.
        prompt_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/annotations/test.json',
        min_video_seconds=8,  
    )
)

val_evaluator = dict(
    type=LightVQAPlus,
    is_gpu=True,
    model_path="metrics/video_quality_assessment/nn_based/lightvqa_plus/Light_VQA_plus/ckpts/last2_SI+TI_epoch_19_SRCC_0.925264.pth",
    swin_weights="metrics/video_quality_assessment/nn_based/lightvqa_plus/Light_VQA_plus/swin_small_patch4_window7_224.pth",
)
