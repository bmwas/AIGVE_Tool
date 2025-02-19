# Copyright (c) IFM Lab. All rights reserved.
# Deprecated

from collections.abc import Iterable, Mapping
from typing import Callable, Optional, Tuple, Union
import datetime, functools, os, subprocess

from torch import distributed as torch_dist
from torch.distributed import ProcessGroup

_LOCAL_PROCESS_GROUP = None


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def init_local_group(node_rank: int, num_gpus_per_node: int):
    """Setup the local process group.

    Setup a process group which only includes processes that on the same
    machine as the current process.

    The code is modified from
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py

    Args:
        node_rank (int): Rank of machines used for training.
        num_gpus_per_node (int): Number of gpus used for training in a single
            machine.
    """  # noqa: W501
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None

    ranks = list(
        range(node_rank * num_gpus_per_node,
              (node_rank + 1) * num_gpus_per_node))
    _LOCAL_PROCESS_GROUP = torch_dist.new_group(ranks)


def get_local_group() -> Optional[ProcessGroup]:
    """Return local process group."""
    if not is_distributed():
        return None

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return _LOCAL_PROCESS_GROUP


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0