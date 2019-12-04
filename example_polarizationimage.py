import cv2
import numpy as np
import os.path
import IMX250MZR
from polarizationAnalyser import GetStokesParameters

def main():
    file_path = "images/polarizer_IMX250MZR.png"
    name, ext = os.path.splitext(file_path)

    bayer = cv2.imread(file_path, -1)
    dtype=bayer.dtype

    img_I = IMX250MZR.demosaicing(bayer)
    upsilon = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])

    polarization = GetStokesParameters(img_I, upsilon)

    maximum_value = np.iinfo(dtype).max if (dtype==np.uint8 or dtype==np.uint16) else np.finfo(dtype).max
    cv2.imwrite(name+"_intensity"+ext, np.clip(polarization.S0 , 0, maximum_value).astype(dtype))
    cv2.imwrite(name+"_max"      +ext, np.clip(polarization.max, 0, maximum_value).astype(dtype))
    cv2.imwrite(name+"_min"      +ext, np.clip(polarization.min, 0, maximum_value).astype(dtype))
    cv2.imwrite(name+"_AoLP"  +".png", cv2.applyColorMap((polarization.AoLP/np.pi*255).astype(np.uint8), cv2.COLORMAP_HSV))
    cv2.imwrite(name+"_DoLP"  +".png", (polarization.DoLP*255).astype(np.uint8))

if __name__=="__main__":
    main()
