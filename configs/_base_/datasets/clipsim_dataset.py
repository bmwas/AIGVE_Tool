# Copyright (c) IFM Lab. All rights reserved.

from mmengine.dataset.sampler import DefaultSampler
from datasets import CLIPSimDataset


val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CLIPSimDataset,
        tokenizer_name='openai/clip-vit-base-patch32',
        video_dir='/storage/drive_1/zizhong/vqa_toolkit/VQA_Toolkit-main/data/toy/evaluate/',
        prompt_dir='/storage/drive_1/zizhong/vqa_toolkit/VQA_Toolkit-main/data/toy/annotations/evaluate.json',
    )
)
