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
        video_dir='/home/xinhao/VQA_Toolkit/aigve/data/toy/evaluate/',
        prompt_dir='/home/xinhao/VQA_Toolkit/aigve/data/toy/annotations/evaluate.json',
        model_name='vgg16',  # User can choose 'vgg16' or 'resnet18'
        max_len=100,
    )
)

val_evaluator = dict(
    type=GSTVQA,
    model_path="metrics/video_quality_assessment/nn_based/gstvqa/GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/training-all-data-GSTVQA-konvid-EXP0-best",
)
