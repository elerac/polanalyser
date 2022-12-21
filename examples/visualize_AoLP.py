"""Visualizing AoLP image with the addition of modulation by DoLP"""
import cv2
import numpy as np
import os
import polanalyser as pa


def main():
    # Read and Demosaicing image
    filepath = "dataset/dragon.png"
    img_raw = cv2.imread(filepath, 0)
    img_demosaiced = pa.demosaicing(img_raw, pa.COLOR_PolarMono_VNG)

    # Calculate Stokes vector
    img_000, img_045, img_090, img_135 = cv2.split(img_demosaiced)

    # Calculate the Stokes vector per-pixel
    image_list = [img_000, img_045, img_090, img_135]
    angles = np.deg2rad([0, 45, 90, 135])
    img_stokes = pa.calcStokes(image_list, angles)

    # Convert Stokes vector to DoLP and AoLP
    img_DoLP = pa.cvtStokesToDoLP(img_stokes)  # 0~1
    img_AoLP = pa.cvtStokesToAoLP(img_stokes)  # 0~pi

    name, ext = os.path.splitext(filepath)

    # Apply the HSV color map on the AoLP image
    img_AoLP_color = pa.applyColorToAoLP(img_AoLP)
    cv2.imwrite(name + "_AoLP" + ".png", img_AoLP_color)

    # Set saturation to DoLP
    # As a result, the image is lighter and has color only at the polarized area
    img_AoLP_light = pa.applyColorToAoLP(img_AoLP, saturation=img_DoLP)
    cv2.imwrite(name + "_AoLP_saturation" + ".png", img_AoLP_light)

    # Set value to DoLP
    # As a result, the image is darker and has color only at the polarized area
    img_AoLP_dark = pa.applyColorToAoLP(img_AoLP, value=img_DoLP)
    cv2.imwrite(name + "_AoLP_value" + ".png", img_AoLP_dark)


if __name__ == "__main__":
    main()
