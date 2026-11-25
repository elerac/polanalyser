"""Blind IMX250MYR color Stokes reconstruction with optional ONNX Runtime."""

import hashlib
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from ._raw2stokes_weights import MODEL_FILES, MODEL_REPO, MODEL_REVISION


def _model_path(precision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    filename, expected = MODEL_FILES[precision]
    args = dict(repo_id=MODEL_REPO, filename=filename, revision=MODEL_REVISION)
    try:
        path = hf_hub_download(**args, local_files_only=True)
    except LocalEntryNotFoundError:
        path = hf_hub_download(**args)

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"pa.raw2stokes() model checksum mismatch for {path}. Remove the damaged cache entry and retry.")
    return path


def _select(available, device, precision):
    gpu = next((ep for ep in ("CUDAExecutionProvider", "CoreMLExecutionProvider") if ep in available), None)
    if device == "gpu" and gpu is None:
        raise RuntimeError("No supported GPU provider for pa.raw2stokes(). On NVIDIA install polanalyser[inference-cuda] "
                           "with matching CUDA/cuDNN libraries. On Apple Silicon install polanalyser[inference].")
    provider = gpu if device != "cpu" and gpu else "CPUExecutionProvider"
    if precision == "auto":
        precision = "fp32" if provider == "CPUExecutionProvider" else "mixed_fp16"
    if provider == "CPUExecutionProvider" and precision == "mixed_fp16":
        raise ValueError("mixed_fp16 requires a supported GPU. Use fp32 for CPU inference.")
    return provider, precision


class _ProviderUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _session(provider, precision, coreml_shape):
    import onnxruntime as ort

    path = _model_path(precision)
    settings = ort.SessionOptions()
    options = {}
    if provider == "CUDAExecutionProvider":
        # A second full-frame allocation plan can exhaust VRAM on display GPUs.
        settings.enable_mem_pattern = False
        ort.preload_dlls()
        # Limit cuDNN scratch allocations for full-resolution mosaics.
        options = {
            "use_tf32": "0",
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        }
    elif provider == "CoreMLExecutionProvider":
        for axis, size in zip(("height", "width"), coreml_shape):
            settings.add_free_dimension_override_by_name(axis, size)
        options = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}
    session = ort.InferenceSession(path, sess_options=settings, providers=[(provider, options)])
    session.disable_fallback()
    if provider not in session.get_providers():
        raise _ProviderUnavailable(f"ONNX Runtime could not initialize {provider}. Check its runtime dependencies.")
    return session


def raw2stokes(raw, /, *, device="auto", precision="auto"):
    """Reconstruct color Stokes vectors from a single IMX250MYR RAW mosaic.

    Parameters
    ----------
    raw : array_like
        Floating-point mosaic of shape (height, width), at least 4 x 4.
        Divide sensor codes by the sensor white level (4095 for right-aligned
        12-bit RAW) to obtain finite values in [0, 1]. Preserve the captured
        black level, white balance and native sensor orientation. For crops,
        both origin coordinates must be multiples of four.
    device : {"auto", "cpu", "gpu"}, optional
        By default, prefer CUDA or Core ML when available, then CPU.
        Explicit "gpu" requires a supported GPU provider.
    precision : {"auto", "fp32", "mixed_fp16"}, optional
        By default, use mixed FP16 on GPU and FP32 on CPU.
        Explicit "mixed_fp16" requires a supported GPU provider.

    Returns
    -------
    stokes : np.ndarray
        Linear float32 array of shape (height, width, 3, 3). The last two
        axes index color (B, G, R) and Stokes component (S0, S1, S2).

    Notes
    -----
    Inference uses the RAW mosaic alone. The model for the selected
    precision is downloaded if it is missing from the local cache.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or min(raw.shape) < 4:
        raise ValueError("Expected a 2-D IMX250MYR mosaic of at least 4 pixels in each dimension.")
    if raw.dtype.kind != "f":
        raise TypeError("Expected floating-point RAW normalized by the sensor white level.")
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError("RAW must contain finite normalized samples in [0, 1].")
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'.")
    if precision not in ("auto", "fp32", "mixed_fp16"):
        raise ValueError("precision must be 'auto', 'fp32', or 'mixed_fp16'.")
    try:
        import onnxruntime as ort
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise ImportError('Install "polanalyser[inference]" or, for NVIDIA CUDA, '
                          '"polanalyser[inference-cuda]" to use raw2stokes.') from exc

    provider, chosen_precision = _select(ort.get_available_providers(), device, precision)
    coreml_shape = raw.shape if provider == "CoreMLExecutionProvider" else None
    try:
        session = _session(provider, chosen_precision, coreml_shape)
    except _ProviderUnavailable:
        if device != "auto" or precision == "mixed_fp16" or provider == "CPUExecutionProvider":
            raise
        warnings.warn(f"{provider} could not initialize. Using CPU FP32.", RuntimeWarning, stacklevel=2)
        session = _session("CPUExecutionProvider", "fp32", None)
    return session.run(["stokes"], {"raw": np.ascontiguousarray(raw, dtype=np.float32)})[0]
