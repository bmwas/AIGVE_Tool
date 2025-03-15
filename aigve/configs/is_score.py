# Copyright (c) IFM Lab. All rights reserved.
from metrics.video_quality_assessment.distribution_based.is_score import ISScore
from mmengine.dataset import DefaultSampler
from datasets import ToyDataset

model = dict(
    type=ISScore,
    model_name='inception_v3',
    input_shape=(299, 299, 3),
    splits=10
)

val_evaluator = dict(
    type=ISScore,
    model_name='inception_v3',  # The model used for IS calculation (InceptionV3)
    input_shape=(299, 299, 3),  # Image input size for InceptionV3
    splits=10  # Number of splits to use when calculating the score
)

val_cfg = dict(
    type='ValLoop'
)

val_dataloader = dict(
    batch_size=8,  # IS evaluation typically processes batches of images
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ToyDataset,
        data_root='AIGVE_Tool/aigve/data/toy/evaluate',
        ann_file='AIGVE_Tool/aigve/data/toy/annotations/evaluate.json',
        modality=dict(use_video=True, use_text=False, use_image=False)
    )
)