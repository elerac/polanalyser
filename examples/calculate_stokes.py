"""Calculate Stokes vector and convert to DoLP and AoLP values.
"""
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

    # Calculate the Stokes vector per-pixel
    angles = np.deg2rad([0, 45, 90, 135])
    img_stokes = pa.calcStokes(img_demosaiced, angles)

    # Decomposition into 3 components (S0, S1, S2)
    img_S0, img_S1, img_S2 = cv2.split(img_stokes)

    # Convert Stokes vector to meaningful values
    img_intensity = pa.cvtStokesToIntensity(img_stokes)
    img_DoLP = pa.cvtStokesToDoLP(img_stokes) # 0~1
    img_AoLP = pa.cvtStokesToAoLP(img_stokes) # 0~pi

    # Convert images to uint8 in order to export
    img_S0_u8 = pa.applyColorToStokes(img_S0, gain=0.5)
    img_S1_u8 = pa.applyColorToStokes(img_S1, gain=8.0)
    img_S2_u8 = pa.applyColorToStokes(img_S2, gain=8.0)

    img_intensity_u8 = adjust_gamma(np.clip(img_intensity, 0, 255).astype(np.uint8), gamma=1/2.2)
    img_DoLP_u8 = np.clip(img_DoLP*255, 0, 255).astype(np.uint8)
    img_AoLP_u8 = pa.applyColorToAoLP(img_AoLP)  # apply pseudo-color

    # Export images
    name, ext = os.path.splitext(filepath)
    cv2.imwrite(f"{name}_S0.png", img_S0_u8)
    cv2.imwrite(f"{name}_S1.png", img_S1_u8)
    cv2.imwrite(f"{name}_S2.png", img_S2_u8)
    cv2.imwrite(f"{name}_intensity.png", img_intensity_u8)
    cv2.imwrite(f"{name}_DoLP.png", img_DoLP_u8)
    cv2.imwrite(f"{name}_AoLP.png", img_AoLP_u8)

if __name__=="__main__":
    main()