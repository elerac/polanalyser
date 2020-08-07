import cv2
import numpy as np
import os.path
import argparse
import polanalyser as pa

def imnormarize(image, max_src, max_dst, dtype):
    image = image.astype(np.float128)
    return np.clip(image/max_src*max_dst, 0, max_dst).astype(dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="bayer image path")
    args = parser.parse_args()

    #Read file
    file_path = args.file_path
    name, ext = os.path.splitext(file_path)
    bayer = cv2.imread(file_path, -1)
    if bayer is None:
        print("No image found")
        exit()

    #Demosaicing
    images = pa.IMX250MZR.demosaicing(bayer)

    #Calculate polarization parameters
    radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    img_stokes = pa.calcStokes(images, radians)

    #Image info
    dtype=bayer.dtype
    max_val = np.iinfo(dtype).max if (dtype==np.uint8 or dtype==np.uint16) else np.finfo(dtype).max
    
    #Convert polarization parameters to images
    S0, S1, S2 = cv2.split(img_stokes)
    img_intensity = S0
    img_Imax = pa.cvtStokesToImax(img_stokes)
    img_Imin = pa.cvtStokesToImin(img_stokes)
    img_AoLP = pa.cvtStokesToAoLP(img_stokes)
    img_DoLP = pa.cvtStokesToDoLP(img_stokes)
    img_AoLP_DoLP = pa.applyLightColorToAoLP(img_AoLP, img_DoLP)
    #img_specular = np.clip((pol.Imax-pol.Imin)/2.0, 0, maximum_value).astype(dtype)

    #Export images
    cv2.imwrite(name+"_intensity"+ext, imnormarize(img_intensity, max_val, max_val, dtype))
    cv2.imwrite(name+"_max"      +ext, imnormarize(img_Imax, max_val, max_val, dtype))
    cv2.imwrite(name+"_min"      +ext, imnormarize(img_Imin, max_val, max_val, dtype))
    #cv2.imwrite(name+"_AoLP"  +".png", imnormarize(img_AoLP, np.pi, 255, np.uint8))
    cv2.imwrite(name+"_DoLP"  +".png", imnormarize(img_DoLP, 1, 255, np.uint8))
    cv2.imwrite(name+"_AoLP+DoLP"  +".png", img_AoLP_DoLP)
    #cv2.imwrite(name+"_specular"+".png", img_specular)

if __name__=="__main__":
    main()
