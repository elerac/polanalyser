"""
Polarization demosaicing example

python demosaicing.py "filepath"
"""

import cv2
import argparse
import os
import polanalyser as pa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="raw image file path (of IMX250MZR)")
    args = parser.parse_args()
    
    filepath = args.filepath
    
    # Read RAW polarization image
    img_raw = cv2.imread(filepath, -1)
    
    if img_raw is None:
        print("{} not found".format(filepath))
        exit()

    # Demosaicing
    img_demosaiced = pa.IMX250MZR.demosaicing(img_raw)

    img_0, img_45, img_90, img_135 = cv2.split(img_demosaiced)
  
    print("Export demosaicing images : {}".format(filepath))
    name, ext = os.path.splitext(filepath)
    cv2.imwrite(name+"-0"  +ext, img_0)
    cv2.imwrite(name+"-45" +ext, img_45)
    cv2.imwrite(name+"-90" +ext, img_90)
    cv2.imwrite(name+"-135"+ext, img_135)

if __name__=="__main__":
    main()
