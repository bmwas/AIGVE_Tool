# Copyright (c) IFM Lab. All rights reserved.

# from transforms import LoadVideoFromFile
from mmengine.dataset.sampler import DefaultSampler
from datasets import Test_VQADataset

# eva_pipeline = [
#     dict(type=LoadVideoFromFile, height = -1, width = -1)
# ]



val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=Test_VQADataset,
        featureaa_dir = '',
        index=None,
        max_len=500,
        feat_dim=2944,
        scale=1,
        # data_root='data/toy/',
        # ann_file='annotations/evaluate.json',
        # data_prefix=dict(video='evaluate'),
        # pipeline=eva_pipeline,
        # backend_args=None,
        # modality=dict(use_video=True, use_text=True, use_image=False),
        # image_frame=None,
        # test_mode=True,
    )
)
