# Copyright (c) IFM Lab. All rights reserved.


from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import os
from os import path as osp

from core.registry import DATASETS
# from mmengine.dataset import BaseDataset
from torch.utils.data import Dataset
import numpy as np
import torch

import scipy.io as scio
from torchvision import transforms

from metrics.video_quality_assessment.nn_based.modular_bvqa.ModularBVQA.extract_laplacian.lp_features import exit_folder, VideoDataset, get_features
from metrics.video_quality_assessment.nn_based.modular_bvqa.ModularBVQA.extract_motion.extract_SlowFast_feature_others import slowfast, pack_pathway_output
from metrics.video_quality_assessment.nn_based.modular_bvqa.ModularBVQA.extract_motion.motion_data_loader import VideoDataset_NR_SlowFast_feature

from metrics.video_quality_assessment.nn_based.gstvqa import Test_VQADataset
import h5py


@DATASETS.register_module()
class KONVID1KDataset_ModularBVQA(Dataset):
    """Dataset used in KONVID1KDataset
    Datails in: https://database.mmsp-kn.de/konvid-1k-database.html

    Args:
        video_dir (str): Path to the video directory.
        metadata_dir (str): Path to the metadata directory.
        database (str): Name of the database. Default is 'KoNViD-1k'.
        num_levels (int): Number of levels in the feature hierarchy.
        layer (int): Layer for feature extraction.
        frame_batch_size (int): Batch size for frames.
        rep_dir (str): Directory for representation data.
        datainfo_path (str): Path to dataset information file.
        save_folder (str): Folder to save extracted features.
        imgs_dir (str): Directory containing image frames.
        resize (int): Resize dimension for frames.
        num_frame (int): Number of frames per video.
        num_workers (int): Number of workers for DataLoader.
        feature_save_folder (str): Directory to save extracted motion features.
    """

    def __init__(self,
                 video_dir='',
                 metadata_dir='',
                 database='KoNViD-1k',
                 num_levels=6,
                 layer=2,
                 frame_batch_size=10,
                 rep_dir='',
                 datainfo_path='data/KoNViD-1k_data.mat',
                 save_folder='/data/konvid_1k/konvid1k_LP_ResNet18/',
                 imgs_dir='data/konvid1k_image_all_fps1',
                 resize=224,
                 num_frame=32,
                 num_workers=2,
                 feature_save_folder=''):
        super().__init__()

        self.video_dir = video_dir
        self.metadata_dir = metadata_dir
        self.database = database
        self.num_levels = num_levels
        self.layer = layer
        self.frame_batch_size = frame_batch_size
        self.rep_dir = rep_dir
        self.datainfo_path = datainfo_path
        self.save_folder = os.getcwd() + save_folder
        self.imgs_dir = imgs_dir
        self.resize = resize
        self.num_frame = num_frame
        self.num_workers = num_workers
        self.feature_save_folder = feature_save_folder

        # Prepare Spatial Rectifier
        dataInfo = scio.loadmat(os.getcwd() + self.rep_dir + self.datainfo_path)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0].split('_')[0])
        video_length_read = 8

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = VideoDataset(
            database_name=self.database, 
            videos_dir=os.getcwd()+self.rep_dir+self.imgs_dir, 
            video_names=video_names, 
            num_levels=self.num_levels
        )

        for i in range(len(dataset)):
            print(i)
            current_data, video_name_str = dataset[i]
            features = get_features(current_data, self.layer, self.frame_batch_size, device)
            exit_folder(os.path.join(save_folder, video_name_str))
            for j in range(video_length_read):
                img_features = features[j*(self.num_levels-1) : (j+1)*(self.num_levels1)]
                np.save(os.path.join(save_folder, video_name_str, '{:03d}'.format(j)), img_features.to('cpu').numpy())
        
        # Prepare Temporal Rectifier
        transformations = transforms.Compose([transforms.Resize([self.resize, self.resize]), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                               std=[0.225, 0.225, 0.225])])
        trainset = VideoDataset_NR_SlowFast_feature(
            database_name=self.database, 
            data_dir=os.getcwd()+self.video_dir, 
            filename_path=os.getcwd()+self.rep_dir+self.datainfo_path, 
            transform=transformations, 
            resize=self.resize, 
            num_frame=self.num_frame)
       
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                               shuffle=False, num_workers=self.num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = slowfast()
        model = model.to(device)
        # do validation after each epoch
        with torch.no_grad():
            model.eval()
            for i, (video, mos, video_name) in enumerate(train_loader):
                video_name = video_name[0]
                print(video_name)
                if not os.path.exists(os.getcwd()+self.feature_save_folder + video_name):
                    os.makedirs(os.getcwd()+self.feature_save_folder + video_name)

                for idx, ele in enumerate(video):
                    # ele = ele.to(device)
                    ele = ele.permute(0, 2, 1, 3, 4)
                    inputs = pack_pathway_output(
                        frames=ele, 
                        device=device)
                    slow_feature, fast_feature = model(inputs)
                    np.save(os.getcwd()+self.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_slow_feature',
                            slow_feature.to('cpu').numpy())
                    np.save(os.getcwd()+self.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature',
                            fast_feature.to('cpu').numpy())

        # Prepare Base Quality Predictor
        dataInfo = scio.loadmat(filename_path)
        n_video = len(dataInfo['video_names'])
        video_names = []

        for i in range(n_video):
            video_names.append(dataInfo['video_names'][i][0][0])

        save_folder = 'data/konvid1k_image_all_fps1'
        for i in range(n_video):
            video_name = video_names[i]
            v_name = video_name.split('_')[0]+'.mp4'
            print('start extract {}th video: {}'.format(i, video_name))
            extract_frame(videos_dir, v_name, save_folder)

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx) -> Any:
        return self.dataset.__getitem__(idx)
        