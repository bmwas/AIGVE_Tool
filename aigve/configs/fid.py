# Copyright (c) IFM Lab. All rights reserved.
from aigve.metrics.video_quality_assessment.distribution_based.fid import FIDScore
from mmengine.dataset import DefaultSampler
from aigve.datasets import ToyDataset

model = dict(
    type=FIDScore,
    model_name='inception_v3',
    input_shape=(299, 299, 3),
    pooling='avg'
)

val_evaluator = dict(
    type=FIDScore,
    model_name='inception_v3',  # The model used for FID calculation (commonly InceptionV3)
    input_shape=(299, 299, 3),  # Image input size for InceptionV3
    pooling='avg'
)

val_cfg = dict(
    type='ValLoop'
)

val_dataloader = dict(
    batch_size=8,  # FID evaluation typically processes batches of images
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ToyDataset,
        real_images_dir='AIGVE_Tool/aigve/data/toy/evaluate/A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-0.mp4',  # Directory of real images
        generated_images_dir='AIGVE_Tool/aigve/data/toy/evaluate/A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-0.mp4'  # Directory of generated images
    )
)
