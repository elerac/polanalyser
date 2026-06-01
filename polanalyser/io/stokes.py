"""EXR file I/O utilities for Stokes image data.

This module provides functions for reading and writing OpenEXR (EXR) image files
containing Stokes polarimetric data in two layouts:

- **Color (BGR)**: shape ``(H, W, 3, 3)`` or ``(H, W, 3, 4)`` — spatial dims ×
  BGR color channels × Stokes components. EXR channels are named ``S*.{R,G,B}``.
- **Mono (grayscale)**: shape ``(H, W, 3)`` or ``(H, W, 4)`` — spatial dims ×
  Stokes components. EXR channels are named ``S*.Y``.

All layouts also store a "base" display channel (``R``, ``G``, ``B`` for color;
``Y`` for mono) populated from the S0 component for convenient EXR viewer display.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

STOKES_ORDER = ("S0", "S1", "S2", "S3")
STOKES_LINEAR_ORDER = STOKES_ORDER[:3]
COLOR_ORDER = ("B", "G", "R")
FILE_COLOR_ORDER = ("R", "G", "B")
MONO_CHANNEL = "Y"


def imread_stokes(filename: str | Path) -> npt.NDArray[np.float32]:
    """Load a multichannel EXR file into a Stokes tensor.

    Auto-detects whether the EXR file stores mono (``S*.Y``) or color
    (``S*.{R,G,B}``) Stokes channels and returns the appropriate array shape.
    EXRs may contain either linear-only ``S0`` through ``S2`` Stokes channels
    or full ``S0`` through ``S3`` Stokes channels.

    Parameters
    ----------
    filename : str or Path
        Path to an EXR image with Stokes channels named ``S*.Y`` (mono) or
        ``S*.{R,G,B}`` (color).

    Returns
    -------
    np.ndarray
        - Color EXR: shape ``(H, W, 3, 3)`` or ``(H, W, 3, 4)`` — BGR color
          planes along axis 2, Stokes components along axis 3.
        - Mono EXR:  shape ``(H, W, 3)`` or ``(H, W, 4)`` — Stokes components
          along axis 2.

    Raises
    ------
    RuntimeError
        If the EXR pixel buffer cannot be read.
    KeyError
        If the required Stokes channels are not found in the file.
    ValueError
        If both mono and color Stokes channels are present (ambiguous format).
    """
    import OpenImageIO as oiio

    filename = str(filename)

    buf = oiio.ImageBuf(filename)
    spec = buf.spec()
    pixels = buf.get_pixels(oiio.FLOAT)  # shape: (height, width, nchannels)

    if pixels is None:
        raise RuntimeError(f"Failed to read pixels from {filename}")

    # Determine format by scanning which Stokes channel sets are present.
    expected_color_linear = {f"{s}.{c}" for s in STOKES_LINEAR_ORDER for c in COLOR_ORDER}
    expected_color_full = {f"{s}.{c}" for s in STOKES_ORDER for c in COLOR_ORDER}
    expected_mono_linear = {f"{s}.{MONO_CHANNEL}" for s in STOKES_LINEAR_ORDER}
    expected_mono_full = {f"{s}.{MONO_CHANNEL}" for s in STOKES_ORDER}

    found_color: set[str] = set()
    found_mono: set[str] = set()
    stokes_set = set(STOKES_ORDER)

    for channel in spec.channelnames:
        if "." not in channel:
            continue
        stokes_id, component = channel.split(".", 1)
        if stokes_id not in stokes_set:
            continue
        if component in COLOR_ORDER:
            found_color.add(f"{stokes_id}.{component}")
        elif component == MONO_CHANNEL:
            found_mono.add(f"{stokes_id}.{component}")

    is_color_linear = found_color == expected_color_linear
    is_color_full = found_color == expected_color_full
    is_color = is_color_linear or is_color_full
    is_mono_linear = found_mono == expected_mono_linear
    is_mono_full = found_mono == expected_mono_full
    is_mono = is_mono_linear or is_mono_full

    if is_color and is_mono:
        raise ValueError(f"Ambiguous EXR: contains both S*.Y and S*.{{R,G,B}} channels in {filename}")
    if not is_color and not is_mono:
        missing = sorted((expected_color_full | expected_mono_full) - found_color - found_mono)
        raise KeyError(f"Missing channels in {filename}: {missing}")

    if is_color:
        stokes_order = STOKES_ORDER if is_color_full else STOKES_LINEAR_ORDER
        stokes_lookup = {name: idx for idx, name in enumerate(stokes_order)}
        color_lookup = {name: idx for idx, name in enumerate(COLOR_ORDER)}
        img_bgr_stokes: npt.NDArray[np.float32] = np.zeros(
            (spec.height, spec.width, len(COLOR_ORDER), len(stokes_order)),
            dtype=np.float32,
        )
        for idx, channel in enumerate(spec.channelnames):
            if "." not in channel:
                continue
            stokes_id, component = channel.split(".", 1)
            if stokes_id in stokes_lookup and component in color_lookup:
                img_bgr_stokes[..., color_lookup[component], stokes_lookup[stokes_id]] = pixels[..., idx]
        return img_bgr_stokes

    # Mono path
    stokes_order = STOKES_ORDER if is_mono_full else STOKES_LINEAR_ORDER
    stokes_lookup = {name: idx for idx, name in enumerate(stokes_order)}
    img_stokes: npt.NDArray[np.float32] = np.zeros((spec.height, spec.width, len(stokes_order)), dtype=np.float32)
    for idx, channel in enumerate(spec.channelnames):
        if "." not in channel:
            continue
        stokes_id, component = channel.split(".", 1)
        if stokes_id in stokes_lookup and component == MONO_CHANNEL:
            img_stokes[..., stokes_lookup[stokes_id]] = pixels[..., idx]
    return img_stokes


def imwrite_stokes(filename: str | Path, img_bgr_stokes: npt.NDArray[np.float32]) -> None:
    """Save a Stokes tensor to an EXR file.

    Accepts either a mono or color Stokes array and writes the appropriate
    channel layout to an EXR file using ``piz`` compression.

    Parameters
    ----------
    filename : str or Path
        Output EXR filepath.  Parent directories are created automatically.
    img_bgr_stokes : np.ndarray
        - Shape ``(H, W, 3)``    — linear-only mono/grayscale Stokes array.
          Written as a ``Y`` base channel plus ``S0.Y``, ``S1.Y``, ``S2.Y``.
        - Shape ``(H, W, 4)``    — mono/grayscale Stokes array.  Written as
          a ``Y`` base channel plus ``S0.Y``, ``S1.Y``, ``S2.Y``, ``S3.Y``.
        - Shape ``(H, W, 3, 3)`` — linear-only BGR color Stokes array. Written
          as ``R``, ``G``, ``B`` base channels plus ``S0..S2.{R,G,B}`` channels.
        - Shape ``(H, W, 3, 4)`` — BGR color Stokes array.  Written as
          ``R``, ``G``, ``B`` base channels plus ``S0..S3.{R,G,B}`` channels.

    Raises
    ------
    ValueError
        If ``img_bgr_stokes`` does not have shape ``(H, W, 3)``, ``(H, W, 4)``,
        ``(H, W, 3, 3)``, or ``(H, W, 3, 4)``.
    RuntimeError
        If the OIIO ``ImageOutput`` cannot be created or opened.
    """
    shape = img_bgr_stokes.shape
    if img_bgr_stokes.ndim == 3 and shape[2] in (len(STOKES_LINEAR_ORDER), len(STOKES_ORDER)):
        mode = "mono"
        stokes_order = STOKES_ORDER if shape[2] == len(STOKES_ORDER) else STOKES_LINEAR_ORDER
    elif img_bgr_stokes.ndim == 4 and shape[2] == len(COLOR_ORDER) and shape[3] in (len(STOKES_LINEAR_ORDER), len(STOKES_ORDER)):
        mode = "color"
        stokes_order = STOKES_ORDER if shape[3] == len(STOKES_ORDER) else STOKES_LINEAR_ORDER
    else:
        raise ValueError(
            "img_bgr_stokes must have shape (H, W, 3) or (H, W, 4) for mono, "
            "or (H, W, 3, 3) or (H, W, 3, 4) for color; "
            f"got {shape!r}"
        )

    import OpenImageIO as oiio

    height, width = img_bgr_stokes.shape[:2]
    channel_planes, channel_names = [], []

    if mode == "mono":
        # Base Y display channel (from S0)
        channel_names.append(MONO_CHANNEL)
        channel_planes.append(img_bgr_stokes[..., 0])
        # Full Stokes tensor channels
        for stokes_idx, stokes_id in enumerate(stokes_order):
            channel_names.append(f"{stokes_id}.{MONO_CHANNEL}")
            channel_planes.append(img_bgr_stokes[..., stokes_idx])
    else:
        stokes_lookup = {name: idx for idx, name in enumerate(stokes_order)}
        # Base RGB display channels (from S0)
        s0_idx = stokes_lookup["S0"]
        for color in FILE_COLOR_ORDER:
            color_idx = COLOR_ORDER.index(color)
            channel_names.append(color)
            channel_planes.append(img_bgr_stokes[..., color_idx, s0_idx])
        # Stokes tensor channels
        for stokes_id in stokes_order:
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
