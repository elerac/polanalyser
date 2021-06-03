"""
Polarization demosaicing example
"""
import os
import cv2
import polanalyser as pa

def main():
    # Read polarization image
    filepath = "dataset/dragon.png"
    img_raw = cv2.imread(filepath, -1)
    
    # Demosaicing
    img_demosaiced = pa.demosaicing(img_raw)

    img_0, img_45, img_90, img_135 = cv2.split(img_demosaiced)
  
    print("Export demosaicing images : {}".format(filepath))
    name, ext = os.path.splitext(filepath)
    cv2.imwrite(f"{name}-000{ext}", img_0)
    cv2.imwrite(f"{name}-045{ext}", img_45)
    cv2.imwrite(f"{name}-090{ext}", img_90)
    cv2.imwrite(f"{name}-135{ext}", img_135)

if __name__=="__main__":
    main()
