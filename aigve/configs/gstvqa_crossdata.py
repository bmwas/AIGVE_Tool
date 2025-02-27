# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler
from metrics.video_quality_assessment.nn_based import GSTVQACrossData
from datasets import GSTVQADatasetCrossData

with read_base():
    # from ._base_.datasets.gstvqa_dataset import *
    from ._base_.default import *

train_index=4

# details: https://github.com/Baoliang93/GSTVQA/blob/8463c9c3e5720349606d8efae7a5aa274bf69e7c/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py#L30
# need to download their dataset first, which is listed in their page
if train_index==1:    
    datainfo_path = "../data/CVD2014info.mat"   
    test_index = [i for i in range(234)]
    model_path = "./models/training-all-data-GSTVQA-cvd14-EXP0-best" 
    feature_path ="../VGG16_mean_std_features/VGG16_cat_features_CVD2014_original_resolution/"
if train_index==2:
    datainfo_path = "../data/LIVE-Qualcomminfo.mat"    
    test_index = [i for i in range(208)]
    model_path = "./models/training-all-data-GSTVQA-liveq-EXP0-best"
    feature_path ="../VGG16_mean_std_features/VGG16_cat_features_LIVE-Qua_1080P/"
if train_index==3:
    datainfo_path = "../data/LIVE_Video_Quality_Challenge_585info.mat" 
    test_index = [i for i in range(585)]
    model_path = "./models/training-all-data-GSTVQA-livev-EXP0-best"
    feature_path ="../VGG16_mean_std_features/VGG_cat_features_LIVE_VQC585_originla_resolution/"
if train_index==4:
    datainfo_path = "data/KoNViD-1kinfo-original.mat" 
    test_index = [i for i in range(1200)]
    model_path = "metrics/video_quality_assessment/nn_based/gstvqa/GSTVQA/TCSVT_Release/GVQA_Release/GVQA_Cross/models/training-all-data-GSTVQA-konvid-EXP0-best"
    feature_path ="metrics/video_quality_assessment/nn_based/gstvqa/GSTVQA/TCSVT_Release/GVQA_Release/VGG16_mean_std_features/VGG16_cat_features_KoNViD_original_resolution/"  

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=GSTVQADatasetCrossData,
        feature_dir=feature_path,
        index=test_index,
        max_len=500,
        feat_dim=2944,
        datainfo_path=datainfo_path,
    )
)

val_evaluator = dict(
    type=GSTVQACrossData,
    metric_path='/metrics/video_quality_assessment/nn_based/gstvqa',
    model_path=model_path,
    datainfo_path=datainfo_path,
    test_index=test_index,
)
