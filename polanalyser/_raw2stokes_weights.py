"""Immutable Hugging Face revision and SHA-256 of each inference artifact."""

MODEL_REPO = "elerac/raw2stokes"
MODEL_REVISION = "fe85d9bc755c0d94952f4a7bbd6ab6301e8d79b2"
MODEL_FILES = {
    "fp32": ("raw2stokes-fp32.onnx", "48a93cbedd626bb6182c81b6e5d78211bbf509048ac4724c169852186da4a1db"),
    "mixed_fp16": ("raw2stokes-mixed_fp16.onnx", "5a895b483be2217d4a6f2142594f6726ee473063104f2ae34479da29882c7d4d"),
}
