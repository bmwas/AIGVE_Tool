# Copyright (c) IFM Lab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Union, Any
from mmengine.evaluator import BaseMetric
from sympy.logic.inference import entails
from transformers import LlamaTokenizer

from core.registry import METRICS
from mmengine.logging import MMLogger
import torch
import torch.nn as nn
import numpy as np
from .mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from .mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

from utils import add_git_submodule, submodule_exists



@METRICS.register_module()
class VideoPhy(BaseMetric):
    def __init__(self,
                hf_token: str,
                collect_device: Optional[Union[str, torch.device]] = None,
                prefix: Optional[str] = None,
                metric_path: str = None,
                model_path: str = 'videophysics/videocon_physics',
                datainfo_path: str = None,
                test_index: int = None,
                 **kwargs):

        """
        This function is used to initialize the VideoPhy metric.

        Args:
            collect_device (str or torch.device): The device to use for collecting the data
            prefix (str): The prefix to use for the metric name
            metric_path (str): The path to the metric
            model_path (str): The path to the model
            datainfo_path (str): The path to the data info
            test_index (int): The index of the test
        """

        super().__init__(collect_device=collect_device, prefix=prefix)
        # self.train_index = train_index
        self.metric_path = metric_path
        self.model_path = model_path
        self.datainfo_path = datainfo_path
        self.test_index = test_index
        self.hf_token = hf_token
        self.results = []

        # self.submodule_path = './metrics/aigve'
        # if not submodule_exists(self.submodule_path):
        #     add_git_submodule(
        #         repo_url='https://github.com/Hritikbansal/videophy.git',
        #         submodule_path=self.submodule_path
        #     )

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, token=self.hf_token)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(self.model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to('cuda')
        self.model.eval()

    def get_entail(self, logits, input_ids):
        """
        This function is used to get the entailment scores.

        Args:
            logits (torch.Tensor): A tensor containing the logits
            input_ids (torch.Tensor): A tensor containing the input IDs
        """
        softmax = nn.Softmax(dim=2)
        logits = softmax(logits)
        token_id_yes = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        token_id_no = self.tokenizer.encode('No', add_special_tokens=False)[0]
        entailment = []
        for j in range(len(logits)):
            for i in range(len(input_ids[j])):
                if input_ids[j][i] == self.tokenizer.pad_token_id:  # pad token if the answer is not present
                    i = i - 1
                    break
                elif i == len(input_ids[j]) - 1:
                    break
            score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
            entailment.append(score)
        entailment = torch.stack(entailment)
        return entailment

    def get_logits(self, data_batch):
        """
        This function is used to get the logits for each input in the data batch.

        Args:
            data_batch (dict): A dictionary containing the data batch
        Returns:
            logits (torch.Tensor): A tensor containing the logits for each input in the data batch
        """
        # Iterate over each item in the data batch
        for k, v in data_batch.items():
            # Check if the item is a tensor
            if torch.is_tensor(v):
                # Convert float tensors to bfloat16
                if v.dtype == torch.float:
                    data_batch[k] = v.bfloat16()
                # Move the tensor to the model's device (e.g., GPU)
                data_batch[k] = data_batch[k].to(self.model.device)

        # print("Data batch: ", data_batch.keys())
        outputs = self.model(pixel_values=data_batch['pixel_values'], video_pixel_values=data_batch['video_pixel_values'],
                        labels=None, \
                        num_images=data_batch['num_images'], num_videos=data_batch['num_videos'], input_ids=data_batch['input_ids'],
                        non_padding_mask=data_batch['non_padding_mask'], \
                        non_media_mask=data_batch['non_media_mask'], prompt_mask=data_batch['prompt_mask'])
        logits = outputs['logits']
        return logits


    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """
        This function is used to process the data batch and compute the metric.

        Args:
            data_batch (dict): A dictionary containing the data batch
            data_samples (list): A list of dictionaries containing the data samples
        """
        logits = self.get_logits(data_batch)
        entails_scores =  self.get_entail(logits, data_batch['input_ids'])

        self.results.extend(entails_scores.cpu().detach().to(torch.float32).numpy().tolist())
        # self.results = entails_scores.cpu().detach().to(torch.float32).numpy().tolist()
        # print(self.results)


    def compute_metrics(self, results: list) -> dict:
        """
        This function is used to compute the metrics.
        
        Args:
            results (list): A list of results
        """
        return {
            'entailment': float(np.mean(results))
        }




