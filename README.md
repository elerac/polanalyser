# PolarizationImaging

## Polarization demosaicing
Demosaic monochrome polarization bayer image taken with the [IMX250MZR](https://www.sony-semicon.co.jp/e/products/IS/polarization/product.html) sensor.
```
$ python3 IMX250MZR.py images/polarizer_IMX250MZR.png
Export demosaicing images : images/polarizer_IMX250MZR.png
images/polarizer_IMX250MZR-0.png
images/polarizer_IMX250MZR-45.png
images/polarizer_IMX250MZR-90.png
images/polarizer_IMX250MZR-135.png
```
OR, use as module.
```python
import cv2
import IMX250MZR

img_bayer = cv2.imread("images/polarizer_IMX250MZR.png", -1)

img_pola = IMX250MZR.demosaicing(img_bayer)

pola0   = img_pola[:,:,0]
pola45  = img_pola[:,:,1]
pola90  = img_pola[:,:,2]
pola135 = img_pola[:,:,3]
```
