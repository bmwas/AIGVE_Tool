# Copyright (c) IFM Lab. All rights reserved.
from .toy_metric import Toy
from fid import FIDScore
from fvd import FVDScore
from is_score import ISScore


__all__ = ['Toy', 'FIDScore', 'FVDScore', 'ISScore']