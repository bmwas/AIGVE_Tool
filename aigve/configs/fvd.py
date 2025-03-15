# Copyright (c) IFM Lab. All rights reserved.
from metrics.video_quality_assessment.distribution_based.fvd import FVDScore
from mmengine.dataset import DefaultSampler
from datasets import ToyDataset

model = dict(
    type=FVDScore,
    model_path='path/to/i3d_model',  # Path to the I3D model for feature extraction
    feature_layer=-2  # Feature layer to extract from the I3D model
)

val_evaluator = dict(
    type=FVDScore,
    model_path='path/to/i3d_model',  # Path to the I3D model
    feature_layer=-2  # Feature layer to extract from the I3D model
)

val_cfg = dict(
    type='ValLoop'
)

val_dataloader = dict(
    batch_size=4,  # FVD evaluation typically processes batches of videos
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ToyDataset,
        real_images_dir='AIGVE_Tool/aigve/data/toy/evaluate/A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-0.mp4', 
        generated_images_dir='AIGVE_Tool/aigve/data/toy/evaluate/A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-0.mp4'
    )
)