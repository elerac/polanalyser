"""Calculate Mueller matrix from images captured under a variety of polarimetric conditions.

Download dataset from here.
- https://drive.google.com/drive/folders/1W66tMue6xi0F1QSG9_sDEgeukV-8QkZs?usp=sharing
"""

import numpy as np
import cv2
import polanalyser as pa


def main():
    path = "dataset/mm3x3_example"
    print(f"Load images from '{path}'")

    images, props = pa.imreadMultiple(path)
    mm_psg = props["mueller_psg"]
    mm_psa = props["mueller_psa"]

    print(mm_psa.shape, mm_psa.dtype)

    img_mueller = pa.calcMueller(images, mm_psg, mm_psa)

    print(img_mueller.shape, img_mueller.dtype)  # (2048, 2448, 3, 3) float64

    # Normalized by a maximum value for visualization
    # Practical Tip: Use np.percentile() not to be affected by outliers.
    img_mueller_maxnorm = np.clip(img_mueller / np.percentile(np.abs(img_mueller), 99), -1, 1)
    img_mueller_maxnorm_vis = pa.applyColorMap(img_mueller_maxnorm, colormap="RdBu", vmin=-1, vmax=1)
    img_mueller_maxnorm_vis_grid = pa.makeGridMueller(img_mueller_maxnorm_vis, border=8)
    cv2.imwrite("mueller_maxnorm_vis_grid.png", img_mueller_maxnorm_vis_grid)

    # Gamma correction for visualization
    img_mueller_gamma = img_mueller_maxnorm
    img_mueller_gamma = pa.gammaCorrection(img_mueller_gamma)
    img_mueller_gamma_vis = pa.applyColorMap(img_mueller_gamma, colormap="RdBu", vmin=-1, vmax=1)
    img_mueller_gamma_vis_grid = pa.makeGridMueller(img_mueller_gamma_vis, border=8)
    cv2.imwrite("mueller_gamma_vis_grid.png", img_mueller_gamma_vis_grid)

    # Normalized by m00 for visualization
    img_mueller_m00norm = img_mueller / img_mueller[..., 0, 0][..., None, None]
    img_mueller_m00norm_vis = pa.applyColorMap(img_mueller_m00norm, colormap="RdBu", vmin=-1, vmax=1)
    img_mueller_m00norm_vis_grid = pa.makeGridMueller(img_mueller_m00norm_vis, border=8)
    cv2.imwrite("mueller_m00norm_vis_grid.png", img_mueller_m00norm_vis_grid)

    # Posi/Nega visualization
    img_mueller_posi_nega = img_mueller_maxnorm * 2
    img_mueller_posi = np.clip(img_mueller_posi_nega, 0, 1)
    img_mueller_nega = np.clip(-img_mueller_posi_nega, 0, 1)
    img_mueller_posi_vis = np.clip(img_mueller_posi * 255, 0, 255).astype(np.uint8)
    img_mueller_nega_vis = np.clip(img_mueller_nega * 255, 0, 255).astype(np.uint8)
    img_mueller_posi_vis_grid = pa.makeGridMueller(img_mueller_posi_vis, border_color=[172, 102, 33], border=32)
    img_mueller_nega_vis_grid = pa.makeGridMueller(img_mueller_nega_vis, border_color=[42, 24, 178], border=32)
    img_mueller_posi_nega_vis_grid = pa.makeGrid([img_mueller_posi_vis_grid, img_mueller_nega_vis_grid], nrow=1, ncol=2, border=0)
    cv2.imwrite("mueller_posi_vis_grid.png", img_mueller_posi_nega_vis_grid)

    # RdBu colorbar
    w, h = 16, 256
    img_bar = np.linspace(1, -1, h)[..., None] * np.ones((1, w))
    img_colorbar = pa.applyColorMap(img_bar, colormap="RdBu", vmin=-1, vmax=1)
    cv2.imwrite("colorbar_RdBu.png", img_colorbar)

    img_bar_gamma = pa.gammaCorrection(img_bar)
    img_colorbar_gamma = pa.applyColorMap(img_bar_gamma, colormap="RdBu", vmin=-1, vmax=1)
    cv2.imwrite("colorbar_gamma_RdBu.png", img_colorbar_gamma)


if __name__ == "__main__":
    main()
