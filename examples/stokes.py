"""
Analysis of Stokes vector
"""
import os
import cv2
import numpy as np
import polanalyser as pa

def convert_u8(image, gamma=1/2.2):
    image = np.clip(image, 0, 255).astype(np.uint8)
    lut = (255.0 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
    return lut[image]

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
    img_Imax = pa.cvtStokesToImax(img_stokes)
    img_Imin = pa.cvtStokesToImin(img_stokes)
    img_DoLP = pa.cvtStokesToDoLP(img_stokes) # 0~1
    img_AoLP = pa.cvtStokesToAoLP(img_stokes) # 0~pi

    # Normarize and save images
    name, ext = os.path.splitext(filepath)
    cv2.imwrite(f"{name}_intensity.png", convert_u8(img_intensity))
    cv2.imwrite(f"{name}_Imax.png",      convert_u8(img_Imax))
    cv2.imwrite(f"{name}_Imin.png",      convert_u8(img_Imin))
    cv2.imwrite(f"{name}_DoLP.png",      convert_u8(img_DoLP*255, gamma=1))
    cv2.imwrite(f"{name}_AoLP.png",      pa.applyColorToAoLP(img_AoLP)) # apply pseudo-color

if __name__=="__main__":
    main()
