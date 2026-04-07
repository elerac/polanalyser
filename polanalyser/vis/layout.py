import numpy as np
import numpy.typing as npt


def tile_mueller(img_mueller: npt.ArrayLike) -> np.ndarray:
    """Tile a Mueller matrix image into a grid layout.

    Parameters
    ----------
    img_mueller : array_like, (H, W, ..., 4, 4) or (4, 4)
        Input Mueller matrix image. The first two axes are interpreted as spatial dimensions and the last two axes as Mueller components.

    Returns
    -------
    np.ndarray, (4H, 4W, ...)
        Tiled Mueller matrix image.
    """
    img_mueller = np.asarray(img_mueller)
    if img_mueller.ndim in [1, 3]:
        raise ValueError(f"Expected img_mueller to have shape (H, W, ..., 4, 4) or (4, 4), got {img_mueller.shape}")

    # Accept a single Mueller matrix input (4, 4) and return as-is.
    if img_mueller.ndim == 2:
        return img_mueller

    # Reshape via a view-only reindexing:
    # (H, W, ..., 4, 4) -> (4, H, 4, W, ...) -> (4H, 4W, ...)
    H, W = img_mueller.shape[0], img_mueller.shape[1]
    mid_axes = list(range(2, img_mueller.ndim - 2))  # the "..." part
    perm = [img_mueller.ndim - 2, 0, img_mueller.ndim - 1, 1, *mid_axes]  # (4, H, 4, W, ...)
    mueller_permuted = img_mueller.transpose(perm)
    img_mueller_tiled = mueller_permuted.reshape((4 * H, 4 * W, *mueller_permuted.shape[4:]))  # (4H, 4W, ...)

    return img_mueller_tiled
