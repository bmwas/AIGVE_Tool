# Copyright (c) IFM Lab. All rights reserved.
from metrics.video_quality_assessment.distribution_based.fid import *
from metrics.video_quality_assessment.distribution_based.is_score import *
from mmengine.dataset import DefaultSampler
from datasets import ToyDataset

val_evaluator = dict(
    type=FIDScore,
    model_name='inception_v3',  # The model used for FID calculation (commonly InceptionV3)
    feature_layer='avg_pool',  # The layer used for feature extraction
    input_size=(299, 299),  # Image input size for InceptionV3
)

val_dataloader = dict(
    batch_size=8,  # FID evaluation typically processes batches of images
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ToyDataset,
        real_images_dir='/home/zhuosheng/VQA_Toolkit/data/toy/evaluate/A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-0.mp4',  # Directory of 
        generated_images_dir='/home/zhuosheng/VQA_Toolkit/data/toy/evaluate/A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-0.mp4',  # Directory of generated images
    )
)
