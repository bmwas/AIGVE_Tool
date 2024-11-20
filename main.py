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
    parser.add_argument("--work-dir", help="the dir to save logs")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

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