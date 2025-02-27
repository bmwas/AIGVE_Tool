import os
import cv2
import json
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset
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

        self.feature_extractor
        self.feature_extractor.eval()

    def forward(self, x):
        x = self.feature_extractor(x)  # Shape: [T, feature_dim, H', W']
        x = torch.flatten(x, start_dim=1)  # Flatten spatial dimensions
        return x  # Shape: [T, feature_dim]


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
        if num_frames < self.max_len:
            pad_frames = torch.zeros((self.max_len - num_frames, *input_frames[0].shape))
            input_frames_tensor = torch.cat((torch.stack(input_frames), pad_frames), dim=0)
        else:
            input_frames_tensor = torch.stack(input_frames[:self.max_len])

        # Convert from [T, H, W, C] to [T, C, H, W]
        input_frames_tensor = input_frames_tensor.permute(0, 3, 1, 2) 

        # Extract features using the chosen model (VGG16 or ResNet)
        with torch.no_grad():
            deep_features = self.feature_extractor(input_frames_tensor)
        
        return deep_features, num_frames
