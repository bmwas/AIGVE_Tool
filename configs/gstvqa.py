# Copyright (c) IFM Lab. All rights reserved.
from mmengine.config import read_base
from metrics.video_quality_assessment.nn_based import GSTVQA

import h5py

with read_base():
    from ._base_.datasets.gstvqa_dataset import *
    from ._base_.default import *

train_index=4

# details: https://github.com/Baoliang93/GSTVQA/blob/8463c9c3e5720349606d8efae7a5aa274bf69e7c/TCSVT_Release/GVQA_Release/GVQA_Cross/cross_test.py#L30
# need to download their dataset first, which is listed in their page
if train_index==1:    
    datainfo_path = "../datas/CVD2014info.mat"   
    test_index = [i for i in range(234)]
    model_path = "./models/training-all-data-GSTVQA-cvd14-EXP0-best" 
    feature_path ="../VGG16_mean_std_features/VGG16_cat_features_CVD2014_original_resolution/"
if train_index==2:
    datainfo_path = "../datas/LIVE-Qualcomminfo.mat"    
    test_index = [i for i in range(208)]
    model_path = "./models/training-all-data-GSTVQA-liveq-EXP0-best"
    feature_path ="../VGG16_mean_std_features/VGG16_cat_features_LIVE-Qua_1080P/"
if train_index==3:
    datainfo_path = "../datas/LIVE_Video_Quality_Challenge_585info.mat" 
    test_index = [i for i in range(585)]
    model_path = "./models/training-all-data-GSTVQA-livev-EXP0-best"
    feature_path ="../VGG16_mean_std_features/VGG_cat_features_LIVE_VQC585_originla_resolution/"
if train_index==4:
    datainfo_path = "../datas/KoNViD-1kinfo-original.mat" 
    test_index = [i for i in range(1200)]
    model_path = "./models/training-all-data-GSTVQA-konvid-EXP0-best"
    feature_path ="../VGG16_mean_std_features/VGG16_cat_features_KoNViD_original_resolution/"  

Info = h5py.File(datainfo_path, 'r') 
scale = Info['scores'][0, :].max()  

val_evaluator = dict(
    type=GSTVQA,
    metric_path='/metrics/video_quality_assessment/nn_based/gstvqa',
    train_index=train_index,
    scale=scale,
    test_index=test_index,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=Test_VQADataset,
        featureaa_dir = feature_path,
        index=test_index,
        max_len=500,
        feat_dim=2944,
        scale=scale,
    )
)