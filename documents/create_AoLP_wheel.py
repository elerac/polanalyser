"""
Create a color map of the AoLP
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_hsv_wheel(filename, is_saturation=False, is_value=False, DoLP_tick=False, dpi=300):
    """
    Plot HSV wheel

    H : AoLP
    S : DoLP or constant value
    V : DoLP or constant value
    
    reference: https://salad-bowl-of-knowledge.github.io/hp/python/2020/01/13/cieluv.html
    """
    def func(r_norm, theta_norm, c, is_saturation=False, is_value=False):
        """
        Function for HSV mapping
        """
        # Calcukate AoLP and DoLP
        AoLP = np.mod(theta_norm*358, 179)
        DoLP = r_norm*255
        # Hue and Saturation and Value
        hue = AoLP
        saturation = DoLP*is_saturation + 255*(1-is_saturation)
        value = DoLP*is_value + 255*(1-is_value)
        return hue*(c==0) + saturation*(c==1) + value*(c==2)
    
    N_theta = 1000
    N_r = 256

    hsv = np.fromfunction(lambda r,theta,c: func(r/(N_r-1), theta/(N_theta-1), c, is_saturation, is_value), (N_r, N_theta, 3), dtype=np.float64).astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)/255.0

    # hue wheel plot
    ax = plt.subplot(111, polar=True)
    if DoLP_tick:
        ax.set_yticks([0, 0.5, 1])
    else:
        ax.set_yticks([])
    
    #get coordinates:
    theta = np.linspace(0, 2*np.pi, rgb.shape[1]+1)
    r = np.linspace(0, 1, rgb.shape[0]+1)
    Theta,R = np.meshgrid(theta, r)
     
    # get color
    color = rgb.reshape((rgb.shape[0]*rgb.shape[1], rgb.shape[2]))
    m = plt.pcolormesh(Theta, R, rgb[:,:,0], color=color, linewidth=0)

    # This is necessary to let the `color` argument determine the color
    m.set_array(None)
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)

    plt.close('all')

def main():
    print("AoLP(Hue) Color Map")
    plot_hsv_wheel("AoLP_wheel.jpg")

    print("AoLP(Hue)+DoLP(Saturation) Color Map")
    plot_hsv_wheel("AoLP_wheel_saturation.jpg", is_saturation=True, DoLP_tick=True)

    print("AoLP(Hue)+DoLP(Value) Color Map")
    plot_hsv_wheel("AoLP_wheel_value.jpg", is_value=True, DoLP_tick=True)

if __name__=="__main__":
    main()
