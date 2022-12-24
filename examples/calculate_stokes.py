"""Calculate Stokes vector and convert to DoLP and AoLP values"""
import os
import argparse
import cv2
import numpy as np
import polanalyser as pa


def adjust_gamma(image, gamma):
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    table = (255.0 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
    return cv2.LUT(image_u8, table)


def generate_colormap(color0, color1):
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[:128] = np.linspace(1, 0, 128)[..., None] * np.array(color0)
    colormap[128:] = np.linspace(0, 1, 128)[..., None] * np.array(color1)
    return np.clip(colormap, 0, 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="dataset/dragon.png")
    args = parser.parse_args()

    # Read image
    filename = args.input
    img_raw = cv2.imread(filename, -1)
    if img_raw is None:
        raise ValueError(f"Input image is None, '{filename}'")

    # Demosaicing
    img_demosaiced = pa.demosaicing(img_raw, pa.COLOR_PolarMono)
    img_000, img_045, img_090, img_135 = cv2.split(img_demosaiced)

    # Calculate Stokes vector from intensity images and polarizer angles
    image_list = [img_000, img_045, img_090, img_135]
    angles = np.deg2rad([0, 45, 90, 135])
    img_stokes = pa.calcStokes(image_list, angles)

    # Stokes to parameters (s0, s1, s2, Intensity(s0) DoLP, AoLP)
    img_s0 = img_stokes[..., 0]
    img_s1 = img_stokes[..., 1]
    img_s2 = img_stokes[..., 2]
    img_intensity = pa.cvtStokesToIntensity(img_stokes)  # same as s0
    img_dolp = pa.cvtStokesToDoLP(img_stokes)  # [0, 1]
    img_aolp = pa.cvtStokesToAoLP(img_stokes)  # [0, pi]

    # Custom colormap (Positive -> Green, Negative -> Red)
    custom_colormap = generate_colormap((0, 0, 255), (0, 255, 0))

    # Apply colormap or adjust the brightness to export images
    img_s0_u8 = pa.applyColorMap(img_s0, "viridis", vmin=0, vmax=np.max(img_s0))
    img_s1_u8 = pa.applyColorMap(img_s1 / img_s0, custom_colormap, vmin=-1, vmax=1)  # normalized by s0
    img_s2_u8 = pa.applyColorMap(img_s2 / img_s0, custom_colormap, vmin=-1, vmax=1)  # normalized by s0
    img_intensity_u8 = adjust_gamma(img_intensity * 0.5, gamma=(1 / 2.2))
    img_dolp_u8 = np.clip(img_dolp * 255, 0, 255).astype(np.uint8)
    img_aolp_u8 = pa.applyColorToAoLP(img_aolp)  # Hue = AoLP, Saturation = 1, Value = 1
    img_aolp_s_u8 = pa.applyColorToAoLP(img_aolp, saturation=img_dolp)  # Hue = AoLP, Saturation = DoLP, Value = 1
    img_aolp_v_u8 = pa.applyColorToAoLP(img_aolp, value=img_dolp)  # Hue = AoLP, Saturation = 1, Value = DoLP

    # Export images
    name, _ext = os.path.splitext(filename)
    cv2.imwrite(f"{name}_s0.png", img_s0_u8)
    cv2.imwrite(f"{name}_s1.png", img_s1_u8)
    cv2.imwrite(f"{name}_s2.png", img_s2_u8)
    cv2.imwrite(f"{name}_intensity.png", img_intensity_u8)
    cv2.imwrite(f"{name}_DoLP.png", img_dolp_u8)
    cv2.imwrite(f"{name}_AoLP.png", img_aolp_u8)
    cv2.imwrite(f"{name}_AoLP_s.png", img_aolp_s_u8)
    cv2.imwrite(f"{name}_AoLP_v.png", img_aolp_v_u8)


if __name__ == "__main__":
    main()
