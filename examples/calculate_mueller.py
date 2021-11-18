"""Calculate Mueller matrix from images captured under a variety of polarimetric conditions.

Download dataset from here.
- https://drive.google.com/drive/folders/1W66tMue6xi0F1QSG9_sDEgeukV-8QkZs?usp=sharing
"""
import cv2
import polanalyser as pa


def main():
    folder_path = "dataset/mueller/various3x3"
    print(f"Load images from '{folder_path}'")
    pimages = pa.io.PolarizationImages(folder_path)
    
    print("Calculate Mueller matrix")
    img_mueller = pa.calcMueller(pimages.image, pimages.mueller_light, pimages.mueller_detector)
    
    print(img_mueller.shape, img_mueller.dtype)

    # Normalized by m00 for visualization
    img_mueller_normalized = img_mueller.copy()
    for i in range(1, img_mueller_normalized.shape[-1]):
        img_mueller_normalized[..., i] /= img_mueller_normalized[..., 0]
    
    filename_visalize = "plot_mueller.png"
    filename_visalize_normalized = "plot_mueller_normalized.png"
    print(f"Visualize and export to '{filename_visalize}' and '{filename_visalize_normalized}'")
    pa.plotMueller(filename_visalize, img_mueller, vabsmax=2.0)
    pa.plotMueller(filename_visalize_normalized, img_mueller_normalized, vabsmax=1.0)

    if img_mueller.shape[-1] == 9:  # 3x3
        img_m00, img_m01, img_m02, \
        img_m10, img_m11, img_m12, \
        img_m20, img_m21, img_m22 = cv2.split(img_mueller)
    elif img_mueller.shape[-1] == 16:  # 4x4
        img_m00, img_m01, img_m02, img_m03, \
        img_m10, img_m11, img_m12, img_m13, \
        img_m20, img_m21, img_m22, img_m23, \
        img_m30, img_m31, img_m32, img_m33, = cv2.split(img_mueller)

if __name__ == "__main__":
    main()