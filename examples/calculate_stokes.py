"""Calculate Stokes vector and convert to DoLP and AoLP values"""
import os
import cv2
import numpy as np
import polanalyser as pa


def adjust_gamma(image, gamma=1.0):
    table = (255.0 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
    return cv2.LUT(image, table)


def main():
    # Read image and demosaicing
    filepath = "dataset/dragon.png"
    img_raw = cv2.imread(filepath, 0)
    img_demosaiced = pa.demosaicing(img_raw)
    img_000, img_045, img_090, img_135 = cv2.split(img_demosaiced)

    # Calculate the Stokes vector per-pixel
    image_list = [img_000, img_045, img_090, img_135]
    angles = np.deg2rad([0, 45, 90, 135])
    img_stokes = pa.calcStokes(image_list, angles)

    # Decomposition into 3 components (s0, s1, s2)
    img_s0, img_s1, img_s2 = cv2.split(img_stokes)

    # Convert Stokes vector to meaningful values
    img_intensity = pa.cvtStokesToIntensity(img_stokes)
    img_dolp = pa.cvtStokesToDoLP(img_stokes)  # 0~1
    img_aolp = pa.cvtStokesToAoLP(img_stokes)  # 0~pi

    # Convert images to uint8 in order to export
    img_s0_u8 = pa.applyColorToStokes(img_s0, gain=0.5)
    img_s1_u8 = pa.applyColorToStokes(img_s1, gain=8.0)
    img_s2_u8 = pa.applyColorToStokes(img_s2, gain=8.0)

    img_intensity_u8 = adjust_gamma(np.clip(img_intensity, 0, 255).astype(np.uint8), gamma=1 / 2.2)
    img_dolp_u8 = np.clip(img_dolp * 255, 0, 255).astype(np.uint8)
    img_aolp_u8 = pa.applyColorToAoLP(img_aolp)  # apply pseudo-color

    # Export images
    name, ext = os.path.splitext(filepath)
    cv2.imwrite(f"{name}_s0.png", img_s0_u8)
    cv2.imwrite(f"{name}_s1.png", img_s1_u8)
    cv2.imwrite(f"{name}_s2.png", img_s2_u8)
    cv2.imwrite(f"{name}_intensity.png", img_intensity_u8)
    cv2.imwrite(f"{name}_DoLP.png", img_dolp_u8)
    cv2.imwrite(f"{name}_AoLP.png", img_aolp_u8)


if __name__ == "__main__":
    main()
