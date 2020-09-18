"""
Visualizing AoLP image
with the addition of modulation by DoLP
"""
import cv2
import numpy as np
import argparse
import os
import polanalyser as pa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="raw image file path (of IMX250MZR)")
    args = parser.parse_args()
    filepath = args.filepath

    # Read and Demosaicing image
    img_raw = cv2.imread(filepath, 0)
    img_demosaiced = pa.demosaicing(img_raw)

    # Calculate Stokes vector
    radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    img_stokes = pa.calcStokes(img_demosaiced, radians)

    # Convert Stokes vector to DoLP and AoLP
    img_DoLP = pa.cvtStokesToDoLP(img_stokes) # 0~1
    img_AoLP = pa.cvtStokesToAoLP(img_stokes) # 0~pi

    name, ext = os.path.splitext(filepath)
    
    # Apply the HSV color map on the AoLP image
    img_AoLP_color = pa.applyColorToAoLP(img_AoLP)
    cv2.imwrite(name+"_AoLP"+".png", img_AoLP_color)
    
    # Set saturation to DoLP
    # As a result, the image is lighter and has color only at the polarized area
    img_AoLP_light = pa.applyColorToAoLP(img_AoLP, saturation=img_DoLP)
    cv2.imwrite(name+"_AoLP_light"+".png", img_AoLP_light)
    
    # Set value to DoLP
    # As a result, the image is darker and has color only at the polarized area
    img_AoLP_dark = pa.applyColorToAoLP(img_AoLP, value=img_DoLP)
    cv2.imwrite(name+"_AoLP_dark"+".png", img_AoLP_dark)

if __name__=="__main__":
    main()
