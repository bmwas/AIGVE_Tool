# Copyright (c) IFM Lab. All rights reserved.

import argparse, logging, os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import torch.multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Toolkit")
    parser.add_argument("config", help="evaluation metric config file path")
    parser.add_argument("--work-dir", help="the dir to save logs")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print("CFG: ", args.config)
    cfg = Config.fromfile(args.config)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # build the runner from config
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.val()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()