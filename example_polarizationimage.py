import cv2
import numpy as np
import os.path
import argparse
import IMX250MZR
from polarizationAnalyser import GetStokesParameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="bayer image path")
    args = parser.parse_args()

    #Read file
    file_path = args.file_path
    name, ext = os.path.splitext(file_path)
    bayer = cv2.imread(file_path, -1)

    #Demosaicing
    img_I = IMX250MZR.demosaicing(bayer)

    #Calculate polarization parameters
    upsilon = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    pol = GetStokesParameters(img_I, upsilon)

    #Image info
    width, height = pol.width, pol.height
    dtype=bayer.dtype
    maximum_value = np.iinfo(dtype).max if (dtype==np.uint8 or dtype==np.uint16) else np.finfo(dtype).max
    
    #Convert polarization parameters to images
    img_intensity = np.clip(pol.S0/2.0 , 0, maximum_value).astype(dtype)
    img_max = np.clip(pol.max, 0, maximum_value).astype(dtype)
    img_min = np.clip(pol.min, 0, maximum_value).astype(dtype)
    img_AoLP = cv2.applyColorMap((pol.AoLP/np.pi*255).astype(np.uint8), cv2.COLORMAP_HSV)
    img_DoLP = np.clip(pol.DoLP*255, 0, 255).astype(np.uint8)
    img_AoLP_DoLP = cv2.cvtColor(np.clip(cv2.merge([pol.AoLP/np.pi*180, pol.DoLP*255, np.ones((height, width))*255]), 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img_specular = np.clip((pol.Imax-pol.Imin)/2.0, 0, maximum_value).astype(dtype)

    #Export images
    cv2.imwrite(name+"_intensity"+ext, img_intensity)
    cv2.imwrite(name+"_max"      +ext, img_max)
    cv2.imwrite(name+"_min"      +ext, img_min)
    cv2.imwrite(name+"_AoLP"  +".png", img_AoLP)
    cv2.imwrite(name+"_DoLP"  +".png", img_DoLP)
    cv2.imwrite(name+"_AoLP+DoLP"  +".png", img_AoLP_DoLP)
    cv2.imwrite(name+"_specular"+".png", img_specular)

if __name__=="__main__":
    main()
