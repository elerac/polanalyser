"""
Calculate Mueller matrix and plot.

Download dataset from
https://drive.google.com/drive/folders/1W66tMue6xi0F1QSG9_sDEgeukV-8QkZs?usp=sharing
"""
import cv2
import numpy as np
import polanalyser as pa

def main():
    print("Read images")
    imlist = []
    degrees_light  = []
    degrees_camera = []
    for deg_l in [0, 45, 90, 135]:
        for deg_c in [0, 45, 90, 135]:
            filename_src = f"dataset/mueller/various_l{deg_l}_c{deg_c}.exr"
            print(f"  {filename_src}")
            img = cv2.imread(filename_src, -1)
            if img is None:
                print("Warning: Image file cannot be opened")
                continue

            imlist.append(img)
            degrees_light.append(deg_l)
            degrees_camera.append(deg_c)
    
    # Convert List to np.ndarray
    # And angles are converted to radians
    images = cv2.merge(imlist)
    radians_light = np.array(degrees_light)*np.pi/180.0
    radians_camera = np.array(degrees_camera)*np.pi/180.0
    
    print("Calculate the Mueller matrix")
    print(f"  images        : {images.shape}")
    print(f"  radians_light : {radians_light.shape}")
    print(f"  radians_camera: {radians_camera.shape}")
    img_mueller = pa.calcMueller(images, radians_light, radians_camera)

    # Decompose the Mueller matrix into its components
    img_m11, img_m12, img_m13,\
    img_m21, img_m22, img_m23,\
    img_m31, img_m32, img_m33  = cv2.split(img_mueller)
    
    print("Plot the Mueller matrix image")
    filename_dst = "plot_mueller.png"
    print(f"  {filename_dst}")
    pa.plotMueller(filename_dst, img_mueller, vabsmax=0.5, dpi=300)

if __name__=="__main__":
    main()
