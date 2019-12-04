import cv2
import numpy as np

def __demosaicing_float(bayer):
    # cv2.cvtColor supports either uint8 or uint16 type
    # float type bayer is demosaiced by this function
    # pros: slow
    # cons: float available
    kernel = np.array([[1/4, 1/2, 1/4],[1/2, 1, 1/2],[1/4, 1/2, 1/4]], dtype=np.float64)
    
    height, width = bayer.shape
    value_position = np.fromfunction(lambda c,y,x: ((y%2)*2+x%2)==c, (4,height,width))
    
    pre_bayer = (np.array((bayer,bayer,bayer,bayer)) * value_position).transpose(1, 2, 0)
    img_polarization = cv2.filter2D(pre_bayer, -1, kernel)
    return img_polarization[:,:,[0, 1, 3, 2]]

def demosaicing(bayer):
    """
    src: bayer image array
    dst: demosaicing image array (0-45-90-135)
    """
    if bayer.dtype!=np.uint8 and bayer.dtype!=np.uint16: return __demosaicing_float(bayer)
    
    BG = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
    GR = cv2.cvtColor(bayer, cv2.COLOR_BayerGR2BGR)
    img_polarization = np.array((BG[:,:,2], GR[:,:,0], BG[:,:,0], GR[:,:,2])).transpose(1, 2, 0)
    return img_polarization

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, nargs="+", help="bayer image file path (of IMX250MZR)")
    args = parser.parse_args()

    for filepath_src in args.filepath:
        img_bayer = cv2.imread(filepath_src, -1)
        
        if img_bayer is None:
            print("{} not found".format(filepath_src))
            continue
           
        img_pola = demosaicing(img_bayer)
      
        print("Export demosaicing images : {}".format(filepath_src))
        name, ext = os.path.splitext(filepath_src)
        for i, upsilon in enumerate([0, 45, 90, 135]):
            pola_u = img_pola[:,:,i]
            filepath_dst = name + "-" + str(upsilon) + ext
            cv2.imwrite(filepath_dst, pola_u)
            print(filepath_dst)
        print()

if __name__=="__main__":
    main()
