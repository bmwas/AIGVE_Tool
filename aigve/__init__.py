# Copyright (c) IFM Lab. All rights reserved.

r"""
This aigve library provides a **comprehensive** and **structured** evaluation framework 
for assessing AI-generated video quality. It integrates multiple evaluation metrics, 
covering diverse aspects of video evaluation, including neural-network-based assessment, 
distribution comparison, vision-language alignment, and multi-faceted analysis.
"""

__version__ = '0.0.1'

# Keep __init__ lightweight; do not import subpackages here.
# Submodules should be imported explicitly, e.g. `from aigve.datasets.fid_dataset import FidDataset`.

__all__ = ['__version__']