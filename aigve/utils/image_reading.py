# Copyright (c) IFM Lab. All rights reserved.

import sys, subprocess
def read_image_detectron2(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    try:
        import detectron2
    except ImportError:
        print("detectron2 is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "detectron2"])

        return detectron2.data.detection_utils.read_image(img_src, format="BGR")