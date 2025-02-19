# Copyright (c) IFM Lab. All rights reserved.
# Deprecated

from mmengine.runner import Runner
# from mmengine.registry import RUNNERS
from core.registry import RUNNERS

from typing import Callable, Dict, List, Optional, Sequence, Union

import torch.nn as nn
from torch.utils.data import DataLoader
from mmengine.optim import OptimWrapper, _ParamScheduler
from mmengine.visualization import Visualizer
from mmengine.hooks import Hook
from mmengine.evaluator import Evaluator
from mmengine.config import Config, ConfigDict

from mmengine.runner import BaseLoop

ConfigType = Union[Dict, Config, ConfigDict]


@RUNNERS.register_module()
class VQARunner(Runner):
    """The Runner for VQA_Toolkit.

    See https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py
    for details.
    """

    def __init__(
        self, 
        model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        auto_scale_lr: Optional[Dict] = None,
        optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: str = 'mmengine',
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        super().__init__(model, work_dir, train_dataloader, val_dataloader, test_dataloader, \
                         train_cfg, val_cfg, test_cfg, auto_scale_lr, optim_wrapper, \
                         param_scheduler, val_evaluator, test_evaluator, default_hooks, \
                         custom_hooks, data_preprocessor, load_from, resume, launcher, \
                         env_cfg, log_processor, log_level, visualizer, default_scope, \
                         randomness, experiment_name, cfg)

        def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
            """Build the VQA_Toolkit evaluation loop.

            Examples of ``loop``:

                # custom validation loop
                loop = dict(type='CustomValLoop')

            Args:
                loop (BaseLoop or dict): A validation loop or a dict to build
                    validation loop. If ``loop`` is a validation loop object, just
                    returns itself.

            Returns:
                :obj:`BaseLoop`: Validation loop object build from ``loop``.
            """
            

            