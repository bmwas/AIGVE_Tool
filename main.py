# Copyright (c) IFM Lab. All rights reserved.

import argparse, logging, os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description="VQA Toolkit")
    parser.add_argument("config", help="evaluation metric config file path")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start the evaluation
    runner.val()


if __name__ == '__main__':
    main()