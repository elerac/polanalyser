import numpy as np
import pytest

import polanalyser as pa


oiio = pytest.importorskip("OpenImageIO")


def _write_exr(path, channel_names, pixels):
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError("Failed to create OIIO ImageOutput")
    spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], len(channel_names), oiio.FLOAT)
    spec.channelnames = tuple(channel_names)
    assert out.open(str(path), spec)
    try:
        out.write_image(np.ascontiguousarray(pixels.astype(np.float32)))
    finally:
        out.close()


def test_imwrite_imread_stokes_mono_roundtrip(tmp_path, stokes_mono):
    path = tmp_path / "mono.exr"

    pa.imwrite_stokes(path, stokes_mono)
    loaded = pa.imread_stokes(path)
    channel_names = oiio.ImageBuf(str(path)).spec().channelnames

    assert loaded.shape == stokes_mono.shape
    assert "Y" in channel_names
    assert "S0.Y" in channel_names
    assert "S3.Y" in channel_names
    np.testing.assert_allclose(loaded, stokes_mono, rtol=0, atol=0)


def test_imwrite_imread_stokes_mono_linear_roundtrip(tmp_path, stokes_mono_linear):
    path = tmp_path / "mono_linear.exr"

    pa.imwrite_stokes(path, stokes_mono_linear)
    loaded = pa.imread_stokes(path)
    channel_names = oiio.ImageBuf(str(path)).spec().channelnames

    assert loaded.shape == stokes_mono_linear.shape
    for channel in ["Y", "S0.Y", "S1.Y", "S2.Y"]:
        assert channel in channel_names
    assert "S3.Y" not in channel_names
    np.testing.assert_allclose(loaded, stokes_mono_linear, rtol=0, atol=0)


def test_imwrite_imread_stokes_color_roundtrip(tmp_path, stokes_color):
    path = tmp_path / "color.exr"

    pa.imwrite_stokes(path, stokes_color)
    loaded = pa.imread_stokes(path)
    channel_names = oiio.ImageBuf(str(path)).spec().channelnames

    assert loaded.shape == stokes_color.shape
    for channel in ["R", "G", "B", "S0.R", "S1.G", "S3.B"]:
        assert channel in channel_names
    np.testing.assert_allclose(loaded, stokes_color, rtol=0, atol=0)


def test_imwrite_imread_stokes_color_linear_roundtrip(tmp_path, stokes_color_linear):
    path = tmp_path / "color_linear.exr"

    pa.imwrite_stokes(path, stokes_color_linear)
    loaded = pa.imread_stokes(path)
    channel_names = oiio.ImageBuf(str(path)).spec().channelnames

    assert loaded.shape == stokes_color_linear.shape
    for channel in ["R", "G", "B"]:
        assert channel in channel_names
    for stokes_id in ["S0", "S1", "S2"]:
        for color in ["R", "G", "B"]:
            assert f"{stokes_id}.{color}" in channel_names
    assert not any(channel.startswith("S3.") for channel in channel_names)
    np.testing.assert_allclose(loaded, stokes_color_linear, rtol=0, atol=0)


def test_imwrite_stokes_rejects_invalid_shape(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        pa.imwrite_stokes(tmp_path / "bad.exr", np.zeros((2, 3, 2), dtype=np.float32))


def test_imread_stokes_rejects_missing_channels(tmp_path):
    path = tmp_path / "missing.exr"
    _write_exr(path, ["Y"], np.zeros((2, 3, 1), dtype=np.float32))

    with pytest.raises(KeyError, match="Missing channels"):
        pa.imread_stokes(path)


def test_imread_stokes_accepts_mono_linear_channels(tmp_path, stokes_mono_linear):
    path = tmp_path / "mono_linear_external.exr"
    channel_names = ["S0.Y", "S1.Y", "S2.Y"]
    _write_exr(path, channel_names, stokes_mono_linear)

    loaded = pa.imread_stokes(path)

    assert loaded.shape == stokes_mono_linear.shape
    np.testing.assert_allclose(loaded, stokes_mono_linear, rtol=0, atol=0)


def test_imread_stokes_accepts_color_linear_channels(tmp_path, stokes_color_linear):
    path = tmp_path / "color_linear_external.exr"
    channel_names = [f"S{stokes_idx}.{color}" for stokes_idx in range(3) for color in ["B", "G", "R"]]
    channel_planes = [
        stokes_color_linear[..., color_idx, stokes_idx]
        for stokes_idx in range(3)
        for color_idx in range(3)
    ]
    _write_exr(path, channel_names, np.stack(channel_planes, axis=-1))

    loaded = pa.imread_stokes(path)

    assert loaded.shape == stokes_color_linear.shape
    np.testing.assert_allclose(loaded, stokes_color_linear, rtol=0, atol=0)


def test_imread_stokes_rejects_ambiguous_channel_layout(tmp_path):
    path = tmp_path / "ambiguous.exr"
    channel_names = [f"S{i}.Y" for i in range(4)]
    channel_names += [f"S{i}.{color}" for i in range(4) for color in ["B", "G", "R"]]
    pixels = np.zeros((2, 3, len(channel_names)), dtype=np.float32)
    _write_exr(path, channel_names, pixels)

    with pytest.raises(ValueError, match="Ambiguous EXR"):
        pa.imread_stokes(path)
