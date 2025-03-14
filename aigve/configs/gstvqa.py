from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.nn_based.gstvqa.gstvqa_metric import GSTVQA
from datasets import GSTVQADataset

with read_base():
    from ._base_.default import *

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=GSTVQADataset,
        # video_dir='AIGVE_Tool/aigve/data/toy/evaluate/', # it has 16 frames for each video, each frame is [512, 512, 3]
        # prompt_dir='AIGVE_Tool/aigve/data/toy/annotations/evaluate.json',
        video_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/videos_3frame/', # it has 81 frames for each video, each frame is [768, 1360, 3]
        prompt_dir='AIGVE_Tool/aigve/data/AIGVE_Bench/annotations/test.json',
        model_name='vgg16',  # User can choose 'vgg16' or 'resnet18'
        max_len=3,
    )
)

val_evaluator = dict(
    type=GSTVQA,
    model_path="metrics/video_quality_assessment/nn_based/gstvqa/GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/training-all-data-GSTVQA-konvid-EXP0-best",
)
