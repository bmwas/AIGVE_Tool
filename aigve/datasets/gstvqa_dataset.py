import os
import cv2
import json
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from core.registry import DATASETS

class FeatureExtractor(nn.Module):
    """Feature extractor using either VGG16 or ResNet18."""
    def __init__(self, model_name='vgg16'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name.lower() == 'vgg16':
            model = models.vgg16(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(model.features.children())[:-1])  # Remove last pooling layer
            self.feature_dim = 1472  # Matches GSTVQA expected feature size
        elif model_name.lower() == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(model.children())[:-2])  # Remove FC layer
            self.feature_dim = 1472  # Adjust if necessary
        else:
            raise ValueError("Unsupported model. Choose 'vgg16' or 'resnet18'.")

        self.feature_extractor.eval()

    def forward(self, x):
        """Extract features from input frames.

        Args:
            x (torch.Tensor): Shape [T, C, H, W]

        Returns:
            mean_features (torch.Tensor): Shape [T, 1472]
            std_features (torch.Tensor): Shape [T, 1472]
        """
        x = self.feature_extractor(x)  # Shape: [T, feature_dim, H', W']: [10, 512, 32, 32]
        
        # Compute std_features **before** spatial pooling (variance across spatial locations)
        std_features = torch.std(x, dim=(2, 3), keepdim=False)  # Shape: [T, 512]
        
        # Compute mean_features **after** spatial pooling (average pooling to get global features)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Shape: [T, 512, 1, 1]
        mean_features = torch.flatten(x, start_dim=1)  # Shape: [T, 512]

        # Ensure correct feature dimension
        if mean_features.shape[1] > self.feature_dim:
            mean_features = mean_features[:, :self.feature_dim]  # Truncate extra features
            std_features = std_features[:, :self.feature_dim] 
        elif mean_features.shape[1] < self.feature_dim:
            padding = torch.zeros((mean_features.shape[0], self.feature_dim - mean_features.shape[1]), device=x.device)
            mean_features = torch.cat((mean_features, padding), dim=1)
            std_features = torch.cat((std_features, padding), dim=1)  # Pad missing dimensions

        return mean_features, std_features  # Shape: [T, 1472] each: [10, 1472]


@DATASETS.register_module()
class GSTVQADataset(Dataset):
    """Dataset for GSTVQA metric, supports feature extraction using VGG16 or ResNet."""

    def __init__(self, video_dir, prompt_dir, model_name='vgg16', max_len=500):
        super(GSTVQADataset, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.model_name = model_name
        self.max_len = max_len
        self.feature_extractor = FeatureExtractor(model_name=model_name)

        self.prompts, self.video_names = self._read_prompt_videoname()

    def _read_prompt_videoname(self):
        with open(self.prompt_dir, 'r') as reader:
            read_data = json.load(reader)

        prompt_data_list, video_name_list = [], []
        for item in read_data["data_list"]:
            prompt = item['prompt_gt'].strip()
            video_name = item['video_path_pd'].strip()
            prompt_data_list.append(prompt)
            video_name_list.append(video_name)

        return prompt_data_list, video_name_list

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        """
        Returns a tuple of:
            deep_features (torch.Tensor): Shape [max_len, 2944]
                Mean and std features extracted from input frames using the chosen model (VGG16 or ResNet).
                Padded to self.max_len if the number of frames is less.
            num_frames (int): The number of frames in the video.
            video_name (str): The file name for the video.
        """
        video_name = self.video_names[index]
        video_path = os.path.join(self.video_dir, video_name)
        input_frames = []

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened() and frame_count < self.max_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, self.frame_size)
            input_frames.append(torch.tensor(frame).float())
            frame_count += 1

        cap.release()

        # Pad or truncate frames to max_len
        num_frames = len(input_frames)
        # print('num_frames: ', num_frames)
        if num_frames < 30:
            pad_frames = torch.zeros((30 - num_frames, *input_frames[0].shape))
            input_frames_tensor = torch.cat((torch.stack(input_frames), pad_frames), dim=0)
            num_frames = 30 # Force min frames to be 30 (since two att_frams=15(kernel_size) used in GSTVQA)
        elif num_frames < self.max_len:
            pad_frames = torch.zeros((self.max_len - num_frames, *input_frames[0].shape))
            input_frames_tensor = torch.cat((torch.stack(input_frames), pad_frames), dim=0)
        else:
            input_frames_tensor = torch.stack(input_frames[:self.max_len])
        # print('input_frames_tensor: ', input_frames_tensor.shape) # shape: toy data [max_len, H(512), W(512), C(3)]
        
        # Convert from [T, H, W, C] to [T, C, H, W]
        input_frames_tensor = input_frames_tensor.permute(0, 3, 1, 2) 

        # Extract features using the chosen model (VGG16 or ResNet)
        with torch.no_grad():
            mean_features, std_features = self.feature_extractor(input_frames_tensor) # Shape: [T, 1472]: [10, 1472]

        # Concatenate to match GSTVQA expected 2944-dim features
        deep_features = torch.cat((mean_features, std_features), dim=1)  # Shape: [T, 2944]

        # Ensure output shape [max_len, 2944] (pad if needed)
        if deep_features.shape[0] < self.max_len:
            pad_size = self.max_len - deep_features.shape[0]
            padding = torch.zeros((pad_size, 2944), device=deep_features.device)
            deep_features = torch.cat((deep_features, padding), dim=0)
        
        
        return deep_features, num_frames, video_name