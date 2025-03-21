# Copyright (c) IFM Lab. All rights reserved.

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from core.registry import LOOPS
from mmengine.runner import BaseLoop, autocast
from mmengine.logging import HistoryBuffer, print_log
import torch
from torch.utils.data import DataLoader
from mmengine.evaluator import Evaluator
import logging

@LOOPS.register_module()
class AIGVELoop(BaseLoop):
    """Loop for VQA metric evaluation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader)
        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        self.val_loss: Dict[str, HistoryBuffer] = dict()

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        outputs = data_batch
        self.evaluator.process(data_batch=data_batch, data_samples=outputs)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
