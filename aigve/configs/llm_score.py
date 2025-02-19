# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.text_video_alignment.gpt_based import LLMScore


with read_base():
    from ._base_.datasets.toy_dataset import *
    from ._base_.default import *

val_evaluator = dict(
    type= LLMScore,
    openai_key = 'OPENAI_KEY',
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=ToyDataset,
        modality=dict(use_video=False, use_text=True, use_image=True),
        image_frame=1,
    )
)