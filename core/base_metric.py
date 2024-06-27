# Copyright (c) IFM Lab. All rights reserved.
# Deprecated

from abc import ABCMeta, abstractmethod

class BaseMetric(metaclass=ABCMeta):
    """Base metric for all metrics in VQA_Toolkit
    """
    def __init__(self, metric_name=None) -> None:
        super().__init__()
        self.metric_name = metric_name
    
    @abstractmethod
    def evaluate(self, cfg):
        """Apply the metric base on the input from the command arguments
        """
        pass

    @abstractmethod
    def from_command(self, cfg):
        """Apply the metric base on the input from the command line
        """
        pass

    @abstractmethod
    def from_cfg(self, cfg):
        """Apply the metric base on the input from the config files
        """
        pass



