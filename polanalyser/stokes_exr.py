"""EXR file I/O utilities for Stokes image data.

This module provides functions for reading and writing OpenEXR (EXR) image files
containing multichannel Stokes polarimetric data. The Stokes image is organized
as a 4D array with shape (height, width, 3, 4) representing:
  - Spatial dimensions: (height, width)
  - Color channels: (B, G, R)
  - Stokes components: (S0, S1, S2, S3)

Functions support bidirectional conversion between EXR files and NumPy arrays,
with automatic handling of channel naming conventions (S*.{R,G,B} format).
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

STOKES_ORDER = ("S0", "S1", "S2", "S3")
COLOR_ORDER = ("B", "G", "R")
FILE_COLOR_ORDER = ("R", "G", "B")


def imread_stokes(filename: str | Path) -> npt.NDArray[np.float32]:
    """Load a multichannel EXR file into a Stokes tensor.

    Parameters
    ----------
    filename : str or Path
        Path to an EXR image whose channels are ordered as ``S*.{R,G,B}``.

    Returns
    -------
    np.ndarray
        Array of shape ``(H, W, 3, 4)`` containing BGR color planes stacked along
        the third axis and Stokes components ``(S0, S1, S2, S3)`` along the fourth.
    """
    import OpenImageIO as oiio

    filename = str(filename)

    # Read full image and metadata once; channel mapping is resolved below.
    buf = oiio.ImageBuf(filename)
    spec = buf.spec()
    pixels = buf.get_pixels(oiio.FLOAT)  # shape: (height, width, nchannels)

    if pixels is None:
        raise RuntimeError(f"Failed to read pixels from {filename}")

    img_bgr_stokes: npt.NDArray[np.float32] = np.zeros(
        (spec.height, spec.width, len(COLOR_ORDER), len(STOKES_ORDER)),
        dtype=np.float32,
    )

    stokes_lookup = {name: idx for idx, name in enumerate(STOKES_ORDER)}
    color_lookup = {name: idx for idx, name in enumerate(COLOR_ORDER)}

    expected_channels = {f"{s}.{c}" for s in STOKES_ORDER for c in COLOR_ORDER}
    found_channels: set[str] = set()

    # Populate tensor from channels named. Ignore unrelated channels.
    for idx, channel in enumerate(spec.channelnames):
        if "." not in channel:
            continue

        stokes_id, component = channel.split(".", 1)
        if stokes_id in stokes_lookup and component in color_lookup:
            img_bgr_stokes[..., color_lookup[component], stokes_lookup[stokes_id]] = pixels[..., idx]
            found_channels.add(f"{stokes_id}.{component}")

    missing_channels = expected_channels - found_channels
    if missing_channels:
        raise KeyError(f"Missing channels in {filename}: {sorted(missing_channels)}")

    return img_bgr_stokes


def imwrite_stokes(filename: str | Path, img_bgr_stokes: npt.NDArray[np.float32]) -> None:
    """Save a Stokes tensor to an EXR file with ``S*.{R,G,B}`` channels.

    Parameters
    ----------
    filename : str or Path
        Output EXR filepath.
    img_bgr_stokes : np.ndarray
        Array of shape ``(H, W, 3, 4)`` storing BGR planes for each of the four
        Stokes components to be serialized.
    """
    if img_bgr_stokes.ndim != 4 or img_bgr_stokes.shape[2:] != (len(COLOR_ORDER), len(STOKES_ORDER)):
        raise ValueError("img_bgr_stokes must have shape (H, W, 3, 4)")

    import OpenImageIO as oiio

    height, width = img_bgr_stokes.shape[:2]
    stokes_lookup = {name: idx for idx, name in enumerate(STOKES_ORDER)}
    channel_planes, channel_names = [], []

    # Add basic RGB channels using S0 component
    s0_idx = stokes_lookup["S0"]
    for color in FILE_COLOR_ORDER:
        color_idx = COLOR_ORDER.index(color)
        channel_names.append(color)
        channel_planes.append(img_bgr_stokes[..., color_idx, s0_idx])

    # Add full Stokes tensor channels
    for stokes_id in STOKES_ORDER:
        stokes_idx = stokes_lookup[stokes_id]
        for color in FILE_COLOR_ORDER:
            color_idx = COLOR_ORDER.index(color)
            channel_names.append(f"{stokes_id}.{color}")
            channel_planes.append(img_bgr_stokes[..., color_idx, stokes_idx])

    # OIIO expects an interleaved HxWxC float32 array for write_image.
    pixels = np.ascontiguousarray(np.stack(channel_planes, axis=-1).astype(np.float32, copy=False))

    filename = str(filename)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    out = oiio.ImageOutput.create(filename)
    if out is None:
        raise RuntimeError(f"Failed to create ImageOutput for {filename}")

    try:
        spec = oiio.ImageSpec(width, height, len(channel_names), oiio.FLOAT)
        spec.channelnames = tuple(channel_names)
        spec["compression"] = "piz"

        if not out.open(filename, spec):
            raise RuntimeError(f"Failed to open {filename} for writing")

        out.write_image(pixels)
    finally:
        out.close()
