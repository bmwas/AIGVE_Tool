# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.nn_based import ModularBVQA
from datasets import KONVID1KDataset_ModularBVQA

with read_base():
    from ._base_.default import *

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=KONVID1KDataset_ModularBVQA,
        video_dir='/data/konvid_1k/data/',
        metadata_dir='/data/konvid_1k/KoNViD-1k_metadata/',

        database='KoNViD-1k',
        num_levels=6,
        layer=2,
        frame_batch_size=10,
        
        rep_dir='/metrics/video_quality_assessment/nn_based/modular_bvqa/ModularBVQA/',
        datainfo_path = 'data/KoNViD-1k_data.mat',
        save_folder = 'data/konvid1k_LP_ResNet18/',
        imgs_dir = 'data/konvid1k_image_all_fps1',

        resize=224,
        num_frame=32,
    )
)

val_evaluator = dict(
    type=ModularBVQA,
    cfg_path='StarVQA_PLUS/configs/Kinetics/TimeSformer_divST_8x32_224.yaml',
    model_name='vit_base_patch16_224',
)
