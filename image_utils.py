"""image_utils.py -- Laboratory 3 helper module.

Turns an image file of ANY size into a fixed-length 1-D feature vector:

    load -> RGB -> resize -> grayscale -> flatten
"""
import numpy as np
from PIL import Image


def image_to_features(path, size=(64, 64), normalize=False):
    """Convert an image file into a fixed-length grayscale pixel vector.

    Parameters
    ----------
    path : str or path-like
        Location of the image file (jpg, png, ...).
    size : tuple[int, int], default (64, 64)
        Target (width, height) in pixels, applied BEFORE grayscale/flatten.
        Because every image is resized to this size, the output length is
        always ``size[0] * size[1]`` regardless of the input dimensions.
    normalize : bool, default False
        If True return float32 values scaled to [0, 1]; otherwise uint8 0-255.

    Returns
    -------
    numpy.ndarray, shape (size[0] * size[1],)
        Row-major flattened grayscale pixel intensities.

    Notes
    -----
    * Resizing discards fine detail; grayscale discards all colour;
      flattening discards the 2-D neighbourhood structure.
    * Images are converted to 'RGB' first so palette / RGBA / greyscale
      files all follow the same code path.
    """
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or size[0] < 1 or size[1] < 1):
        raise ValueError(f"size must be (width, height) positive ints, got {size!r}")

    with Image.open(path) as img:
        img = img.convert("RGB")                       # consistent 3 channels
        img = img.resize(tuple(size), Image.LANCZOS)   # fixed dimensions
        img = img.convert("L")                         # grayscale (luminance)
        vec = np.asarray(img).flatten()                # 2-D -> 1-D (row-major)

    if normalize:
        return vec.astype(np.float32) / 255.0
    return vec
