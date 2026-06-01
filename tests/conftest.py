import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def stokes_mono():
    s0 = np.array([[1.0, 1.2, 1.4], [1.6, 1.8, 2.0]], dtype=np.float32)
    dop = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=np.float32)
    aolp = np.array([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]], dtype=np.float32)
    eang = np.array([[0.0, 0.05, -0.05], [0.1, -0.1, 0.2]], dtype=np.float32)

    import polanalyser as pa

    return pa.stokes(s0, dop, aolp, eang).astype(np.float32)


@pytest.fixture
def stokes_mono_linear(stokes_mono):
    return stokes_mono[..., :3].copy()


@pytest.fixture
def stokes_color(stokes_mono):
    gains = np.array([0.7, 1.0, 1.3], dtype=np.float32)
    return stokes_mono[:, :, None, :] * gains[None, None, :, None]


@pytest.fixture
def stokes_color_linear(stokes_color):
    return stokes_color[..., :3].copy()


@pytest.fixture
def mueller_image():
    base = np.arange(2 * 3 * 4 * 4, dtype=np.float64).reshape(2, 3, 4, 4)
    return base / base.max()


@pytest.fixture
def tiny_arrays():
    return {
        "gray_u8": np.arange(12, dtype=np.uint8).reshape(3, 4),
        "bgr_u8": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
        "gray_u16": (10 * np.arange(12, dtype=np.uint16)).reshape(3, 4),
        "tensor_f32": np.linspace(0, 1, 2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5),
    }


@pytest.fixture
def assert_angles_close_mod_pi():
    def _assert(actual, expected, *, atol=1e-12):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        diff = (actual - expected + np.pi / 2) % np.pi - np.pi / 2
        np.testing.assert_allclose(diff, 0.0, atol=atol)

    return _assert
