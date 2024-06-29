# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.video_quality_assessment.nn_based import GSTVQA

with read_base():
    from ._base_.datasets.toy_dataset import *
    from ._base_.default import *

val_evaluator = dict(
    type= GSTVQA,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=ToyDataset,
        modality=dict(use_video=True, use_text=False, use_image=False),
        image_frame=None,
    )
)