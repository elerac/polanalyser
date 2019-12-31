# PolarizationImaging

## Polarization demosaicing
Demosaic monochrome polarization bayer image taken with the [IMX250MZR](https://www.sony-semicon.co.jp/e/products/IS/polarization/product.html) sensor.
```
$ python3 IMX250MZR.py images/polarizer_IMX250MZR.png
```
OR, use as a module.
```python
import cv2
import IMX250MZR

img_bayer = cv2.imread("images/polarizer_IMX250MZR.png", -1)

img_pola = IMX250MZR.demosaicing(img_bayer)

img_pola0   = img_pola[:,:,0]
img_pola45  = img_pola[:,:,1]
img_pola90  = img_pola[:,:,2]
img_pola135 = img_pola[:,:,3]
```

## Estimate the Stokes parameters
```python
import cv2
import numpy as np
import IMX250MZR
from polarizationAnalyser import GetStokesParameters

img_bayer = cv2.imread("images/polarizer_IMX250MZR.png", -1)
img_pola = IMX250MZR.demosaicing(img_bayer)

upsilon = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
polarization = GetStokesParameters(img_pola, upsilon)

img_S0 = polarization.S0
img_S1 = polarization.S1
img_S2 = polarization.S2

img_AoLP = polarization.AoLP
img_DoLP = polarization.DoLP
img_Imax = polarization.max
img_Imin = polarization.min
```

|Example of results| | |
|:-:|:-:|:-:|
|Normal (S0)|Imax|Imin|
|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_intensity.jpg)|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_max.jpg)|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_min.jpg)|
|DoLP|AoLP||
|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_DoLP.jpg)|![](https://github.com/elerac/PolarizationImaging/blob/manuscript/polarizer_IMX250MZR_AoLP.jpg)||
