"""Calculate Stokes vector and convert to DoLP and AoLP values"""

import os
import argparse
import cv2
import numpy as np
import polanalyser as pa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="dataset/dragon.png")
    args = parser.parse_args()

    # Read image
    filename = args.input
    img_raw = cv2.imread(filename, -1)
    if img_raw is None:
        raise ValueError(f"Input image is None, '{filename}'")

    # Adjust the brightness of the EXR image
    # assuming the input image is in the range of [0,1].
    ext = os.path.splitext(filename)[1]
    if ext == ".exr":
        img_raw = 255.0 * img_raw

    # Demosaicing
    img_demosaiced_list = pa.demosaicing(img_raw, pa.COLOR_PolarMono)
    img_000, img_045, img_090, img_135 = img_demosaiced_list

    # Calculate Stokes vector from intensity images and polarizer angles
    img_stokes = pa.calcStokes(img_demosaiced_list, np.deg2rad([0, 45, 90, 135]))

    # Stokes to parameters (s0, s1, s2, Intensity(s0) DoLP, AoLP)
    img_s0 = img_stokes[..., 0]
    img_s1 = img_stokes[..., 1]
    img_s2 = img_stokes[..., 2]
    img_intensity = pa.cvtStokesToIntensity(img_stokes)  # same as s0
    img_dolp = pa.cvtStokesToDoLP(img_stokes)  # [0, 1]
    img_aolp = pa.cvtStokesToAoLP(img_stokes)  # [0, pi]

    # Apply colormap or adjust the brightness to export images
    img_intensity_vis = np.clip(255 * ((img_intensity / 255 / 2) ** (1 / 2.2)), 0, 255).astype(np.uint8)
    img_s0_vis = pa.applyColorMap(img_s0, "bwr", vmin=-64, vmax=64)
    img_s1_vis = pa.applyColorMap(img_s1, "bwr", vmin=-64, vmax=64)
    img_s2_vis = pa.applyColorMap(img_s2, "bwr", vmin=-64, vmax=64)
    img_dolp_vis = pa.applyColorToDoP(img_dolp)
    img_aolp_vis = pa.applyColorToAoLP(img_aolp)  # Hue = AoLP, Saturation = 1, Value = 1
    img_aolp_s_vis = pa.applyColorToAoLP(img_aolp, saturation=img_dolp)  # Hue = AoLP, Saturation = DoLP, Value = 1
    img_aolp_v_vis = pa.applyColorToAoLP(img_aolp, value=img_dolp)  # Hue = AoLP, Saturation = 1, Value = DoLP

    # Export images
    name, _ext = os.path.splitext(filename)
    cv2.imwrite(f"{name}_s0.png", img_s0_vis)
    cv2.imwrite(f"{name}_s1.png", img_s1_vis)
    cv2.imwrite(f"{name}_s2.png", img_s2_vis)
    cv2.imwrite(f"{name}_intensity.png", img_intensity_vis)
    cv2.imwrite(f"{name}_DoLP.png", img_dolp_vis)
    cv2.imwrite(f"{name}_AoLP.png", img_aolp_vis)
    cv2.imwrite(f"{name}_AoLP_s.png", img_aolp_s_vis)
    cv2.imwrite(f"{name}_AoLP_v.png", img_aolp_v_vis)


if __name__ == "__main__":
    main()
