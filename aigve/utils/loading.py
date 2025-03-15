# Copyright (c) IFM Lab. All rights reserved.

from typing import Optional

from core.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform

from PIL import Image, ImageSequence
from decord import VideoReader
import numpy as np
import torch


@TRANSFORMS.register_module()
class LoadVideoFromFile(BaseTransform):
    """Load a video from file.

    Required Keys:

        - video_path_pd

    Modified Keys:
        - video_pd

    Args:
        height: int, default is -1
            Desired output height of the video, unchanged if `-1` is specified.
        width: int, default is -1
            Desired output width of the video, unchanged if `-1` is specified.
            See details in: https://github.com/dmlc/decord/blob/master/python/decord/video_reader.py#L18
    """
    
    def __init__(self, height: int = -1, width: int = -1):
        self.height = height
        self.width = width


    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load video. 
        Referred to 'https://github.com/Vchitect/VBench/blob/master/vbench/utils.py#L103'

        The function supports loading video in GIF (.gif), PNG (.png), and MP4 (.mp4) formats.
        Depending on the format, it processes and extracts frames accordingly.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded video in shape (F, C, H, W) and 
            meta information if needed. F is the number of frames, C is the 
            number of channels, H is the height, and W is the width.
        
        Raises:
            - NotImplementedError: If the video format is not supported.
        
        The function first determines the format of the video file by its extension.
        For GIFs, it iterates over each frame and converts them to RGB.
        For PNGs, it reads the single frame, converts it to RGB.
        For MP4s, it reads the frames using the VideoReader class and converts them to NumPy arrays.
        If a data_transform is provided, it is applied to the buffer before converting it to a tensor.
        Finally, the tensor is permuted to match the expected (F, C, H, W) format.
        """

        video_path = results['video_path_pd']
        if video_path.endswith('.gif'):
            frame_ls = []
            img = Image.open(video_path)
            for frame in ImageSequence.Iterator(img):
                frame = frame.convert('RGB')
                frame = np.array(frame).astype(np.uint8)
                frame_ls.append(frame)
            buffer = np.array(frame_ls).astype(np.uint8) # (F, H, W, C), np.uint8
        elif video_path.endswith('.png'):
            frame = Image.open(video_path)
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls = [frame]
            buffer = np.array(frame_ls) # (1, H, W, C), np.uint8
        elif video_path.endswith('.mp4'):
            import decord
            decord.bridge.set_bridge('native')
            if self.width and self.height:
                video_reader = VideoReader(video_path, width=self.width, height=self.height, num_threads=1)
            else:
                video_reader = VideoReader(video_path, num_threads=1)
            frames = video_reader.get_batch(range(len(video_reader)))  # (F, H, W, C), torch.uint8
            buffer = frames.asnumpy().astype(np.uint8) # (F, H, W, C), np.uint8
        else:
            raise NotImplementedError

        frames = torch.Tensor(buffer)
        frames = frames.permute(0, 3, 1, 2) # (F, C, H, W), torch.uint8
        results['video_pd'] = frames

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}, '
                    f'height={self.height}, '
                    f'width={self.width}')