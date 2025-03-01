import os
import cv2
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from core.registry import DATASETS

@DATASETS.register_module()
class SimpleVQADataset(Dataset):
    """
    Dataset for SimpleVQA.
    Each sample returns:
        - spatial_features (torch.Tensor): Extracted spatial frames.
        - motion_features (torch.Tensor): Extracted motion-based clips.
        - video_name (str): Video filename.
    """

    def __init__(self, video_dir, prompt_dir, max_len=8):
        super(SimpleVQADataset, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_dir = video_dir
        self.prompt_dir = prompt_dir
        self.max_len = max_len

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
    
    def video_processing_spatial(self, video_path):
        """
        Extracts spatial frames with proper resizing and normalization.
            - Key frame extraction: It selects 1 frame per second.
            - Standard input size: It resizes frames to 448 * 448 (after an initial resize to 520).
        """
        video_capture = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))

        # Compute the number of total seconds of the video
        video_length_read = int(video_length/video_frame_rate)
        print('video_length_read (s): ', video_length_read)
        transformations = transforms.Compose([
            transforms.Resize(520),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transformed_video = torch.zeros([video_length_read, 3, 448, 448])

        video_read_index = 0
        frame_idx = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # Key frame extraction
                if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):
                    read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    read_frame = transformations(read_frame)
                    transformed_video[video_read_index] = read_frame
                    video_read_index += 1
                frame_idx += 1

        # Fill missing frames if needed
        if video_read_index < video_length_read:
            for i in range(video_read_index, video_length_read):
                transformed_video[i] = transformed_video[video_read_index - 1]

        video_capture.release()
        return transformed_video, video_name

    def video_processing_motion(self, video_path):
        """
        Extracts motion-based clips suitable for SlowFast.
        Processes 8-second clips, extracting 32 frames per clip.
        """
        video_capture = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)

        video_channel = 3
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
        video_clip = min(video_length // video_frame_rate, self.max_len)

        video_clip_min = 8
        video_length_clip = 32

        # Prepare transformation pipeline
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

        transformed_frame_all = torch.zeros([video_length, video_channel, 224, 224])
        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        # Extract motion-based clips
        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, 224, 224])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i * video_frame_rate:(i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
        
        return transformed_video_all, video_name

    def __getitem__(self, index):
        """
        Returns:
            spatial_features (torch.Tensor): Shape [max_len, C, H, W]
            motion_features (List[torch.Tensor]): List of motion feature tensors
            video_name (str): Video filename
        """
        video_name = self.video_names[index]
        video_path = os.path.join(self.video_dir, video_name)

        spatial_features, video_name = self.video_processing_spatial(video_path)
        motion_features, video_name = self.video_processing_motion(video_path)
        print('spatial_features: ', spatial_features.shape)
        print('motion_features: ', len(motion_features))
        print('motion_features[0]: ', motion_features[0].shape)

        return spatial_features, motion_features, video_name
