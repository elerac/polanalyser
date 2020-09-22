"""
Analysis of Stokes vector
"""
import cv2
import numpy as np
import os
import polanalyser as pa

def main():
    # Read image and demosaicing
    filepath = "dataset/dragon.png"
    img_raw = cv2.imread(filepath, 0)
    img_demosaiced = pa.demosaicing(img_raw)

    # Calculate the Stokes vector per-pixel
    radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    img_stokes = pa.calcStokes(img_demosaiced, radians)

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
    dtype_raw = img_raw.dtype

    img_intensity_norm = normarize(img_intensity, dtype_raw)
    cv2.imwrite(name+"_intensity"+ext, img_intensity_norm)
    
    img_Imax_norm = normarize(img_Imax, dtype_raw)
    cv2.imwrite(name+"_Imax"+ext, img_Imax_norm)
    
    img_Imin_norm = normarize(img_Imin, dtype_raw)
    cv2.imwrite(name+"_Imin"+ext, img_Imin_norm)

    img_DoLP_norm = normarize(img_DoLP*255, np.uint8, gamma=1.0)
    cv2.imwrite(name+"_DoLP"+".png", img_DoLP_norm)

    img_AoLP_color = pa.applyColorToAoLP(img_AoLP) # apply pseudo-color
    cv2.imwrite(name+"_AoLP"+".png", img_AoLP_color)

def normarize(image, dtype=np.uint8, gamma=2.2):
    """
    Thie function processes image for exporting.
    1. apply gamma
    2. clip
    3. convert dtype
    """
    max_val = np.iinfo(dtype).max if (dtype==np.uint8 or dtype==np.uint16) else np.finfo(dtype).max
    img_gamma_applied = ((image/max_val)**(1.0/gamma))*max_val
    img_normarized = np.clip(img_gamma_applied, 0, max_val).astype(dtype)
    return img_normarized

if __name__=="__main__":
    main()
