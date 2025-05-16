
from typing import Dict, List, Optional, Sequence, Union, Any

from mmengine.evaluator import BaseMetric

from core.registry import METRICS
import torch
import os
from .vbench_utils import VBenchwithReturn

@METRICS.register_module()
class VbenchMetric(BaseMetric):
    def __init__(self,
                collect_device: Optional[Union[str, torch.device]] = None,
                prefix: Optional[str] = None,
                vbench_prompt_json_path: str = None, eval_aspects: List[str] = None, eval_mode: str = 'vbench_standard',
                local: bool=False, read_frame: bool=False, category:str='', imaging_quality_preprocessing_mode:str='longer', **kwargs):
        """
        Args:
            collect_device (Optional[Union[str, torch.device]]): The device to collect the data on.
            prefix (Optional[str]): The prefix to use for the metric.
            vbench_prompt_json_path (str): The path to the vbench prompt JSON file.
            eval_aspects (list): the evaluation aspects, if the vbench_prompt_json_path is not None, the available aspects are
            ['subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality',
            'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship',
            'scene', 'temporal_style', 'appearance_style', 'overall_consistency'] if the vbench_prompt_json_path is None, the available aspects are ['subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality']
            eval_mode (str): the evaluation mode, if the vbench_prompt_json_path is not None, the available modes are ['vbench_standard', 'vbench_category'] if the vbench_prompt_json_path is None, the available modes are ['custom_input']
            local (bool): whether to use local mode, if True, the model will be loaded locally, if False, the model will be loaded from the internet
            read_frame (bool): whether to read the frame from the video, if True, the model will read the frame from the video, if False, the model will not read the frame from the video
            category(str): The category to evaluate on, usage: --category=animal.
            imaging_quality_preprocessing_mode(str): 1. 'shorter': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
            2. 'longer': if the longer side is more than 512, the image is resized so that the longer side is 512.
            3. 'shorter_centercrop': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
            Then the center 512 x 512 after resized is used for evaluation.
            4. 'None': no preprocessing
        """
        super().__init__(collect_device=collect_device, prefix=prefix)
        # self.train_index = train_index

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.results = []
        self.vbench_prompt_json_path = vbench_prompt_json_path
        self.vbench = VBenchwithReturn(device=self.device, full_info_dir=self.vbench_prompt_json_path)
        self.eval_aspects = eval_aspects
        self.eval_mode = eval_mode
        self.local = local
        self.read_frame = read_frame
        self.category = category
        self.imaging_quality_preprocessing_mode = imaging_quality_preprocessing_mode

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """
        Args:
            data_batch (Any): The data batch to process.
            data_samples (Sequence[dict]): The data samples to process.
        """

        if type(data_batch['video_path']) == list and len(data_batch['video_path']) > 1:
            video_roots = set([os.path.dirname(video_path) for video_path in data_batch['video_path']])
            if len(video_roots) > 1:
                raise ValueError('The video paths should be in the same directory.')
            else:
                video_path = video_roots.pop()
        elif type(data_batch['video_path']) == list and len(data_batch['video_path']) == 1:
            video_path = data_batch['video_path'][0]
        elif type(data_batch['video_path']) == str:
            video_path = data_batch['video_path']
        else:
            raise ValueError('The video paths should be a list or a string.')



        kwargs = {}

        if self.category != '':
            kwargs['category'] = self.category

        kwargs['imaging_quality_preprocessing_mode'] = self.imaging_quality_preprocessing_mode

        result = self.vbench.evaluate(
            videos_path = video_path,
            name = f'results_{self.eval_mode}',
            prompt_list=data_batch['prompt'], # pass in [] to read prompt from filename
            dimension_list = self.eval_aspects,
            local=self.local,
            read_frame=self.read_frame,
            mode=self.eval_mode, **kwargs)


        self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        """
        Args:
            results (list): The results to compute the metrics from.
        """
        print('results:', results)
        # results = np.array(results)
        # mean_scores = np.mean(results, axis=1)
        #
        # return {'visual_quailty': results[:, 0].tolist(),
        #         'temporal_consistency': results[:, 1].tolist(),
        #         'dynamic_degree': results[:, 2].tolist(),
        #         'text-to-video_alignment': results[:, 3].tolist(),
        #         'factual_consistency': results[:, 4].tolist(),
        #         'summary': {'visual_quality': mean_scores[0], 'temporal_consistency': mean_scores[1],
        #                     'dynamic_degree': mean_scores[2], 'text-to-video_alignment': mean_scores[3],
        #                     'factual_consistency': mean_scores[4]}}