# Copyright (c) IFM Lab. All rights reserved.
# Deprecated
import datetime, logging, os, platform, warnings

from mmengine.logging import print_log
from mmengine.utils import digit_version

def setup_cache_size_limit_of_dynamo():
    """Setup cache size limit of dynamo.

    Note: Due to the dynamic shape of the loss calculation and
    post-processing parts in the object detection algorithm, these
    functions must be compiled every time they are run.
    Setting a large value for torch._dynamo.config.cache_size_limit
    may result in repeated compilation, which can slow down training
    and testing speed. Therefore, we need to set the default value of
    cache_size_limit smaller. An empirical value is 4.
    """

    import torch
    if digit_version(torch.__version__) >= digit_version('2.0.0'):
        if 'DYNAMO_CACHE_SIZE_LIMIT' in os.environ:
            import torch._dynamo
            cache_size_limit = int(os.environ['DYNAMO_CACHE_SIZE_LIMIT'])
            torch._dynamo.config.cache_size_limit = cache_size_limit
            print_log(
                f'torch._dynamo.config.cache_size_limit is force '
                f'set to {cache_size_limit}.',
                logger='current',
                level=logging.WARNING)