# Polanalyser
Polanalyser is polarization image analysis tool.

It can be used for 
* [Demosaicing of bayer images taken with a polarization camera](#polarization-demosaicing)
* [Analysis of Stokes vector](#analysis-of-stokes-vector)
* [Analysis of Mueller matrix](#analysis-of-mueller-matrix)

## Requirement
* OpenCV
* Numpy
* matplotlib (optional)

## Installation
```sh
pip install git+https://github.com/elerac/polanalyser
```

## Polarization Image Dataset
A dataset of images taken by a polarization camera (FLIR, BFS-U3-51S5P-C) is available.

[**Click here and download dataset**](https://drive.google.com/drive/folders/1vCe9N05to5_McvwyDqxTmLIKz7vRzmbX?usp=sharing)


## Usage
### Polarization demosaicing
Demosaic raw polarization image taken with the polarization sensor (e.g. [IMX250MZR / MYR](https://www.sony-semicon.co.jp/e/products/IS/polarization/product.html)).
![](documents/demosaicing.png)
```python
import cv2
import polanalyser as pa

img_raw = cv2.imread("dataset/dragon.png", 0)

img_demosaiced = pa.demosaicing(img_raw)

img_0, img_45, img_90, img_135 = cv2.split(img_demosaiced)
```

### Analysis of Stokes vector
The [**Stokes vector**](https://en.wikipedia.org/wiki/Stokes_parameters) describes the polarization states. We can measure these values by using the *linear polarizer* (To measure the circular polarization S3, we also need to use the *retarder*).
![](documents/stokes_setup.png)
Stokes vector can be converted to meaningful values. *Degree of Linear Polarization* (DoLP) represents how much the light is polarized. The value is 1 for perfectly polarized light and 0 for unpolarized light. *Angle of Linear Polarization* (AoLP) represents the polarization angle of the incident light relative to the camera sensor axis. The value ranges from 0 to 180 degrees.
```python
import cv2
import numpy as np
import polanalyser as pa

# Read image and demosaicing
img_raw = cv2.imread("dataset/dragon.png", 0)
img_demosaiced = pa.demosaicing(img_raw)

# Calculate the Stokes vector per-pixel
angles = np.deg2rad([0, 45, 90, 135])
img_stokes = pa.calcStokes(img_demosaiced, angles)

# Decompose the Stokes vector into its components
img_S0, img_S1, img_S2 = cv2.split(img_stokes)

# Convert the Stokes vector to Intensity, DoLP and AoLP
img_intensity = pa.cvtStokesToIntensity(img_stokes)
img_DoLP      = pa.cvtStokesToDoLP(img_stokes)
img_AoLP      = pa.cvtStokesToAoLP(img_stokes)
```

||Example of results | |
|:-:|:-:|:-:|
|Intensity|DoLP|AoLP|
|![](documents/dragon_IMX250MZR_intensity.jpg)|![](documents/dragon_IMX250MZR_DoLP.jpg)|![](documents/dragon_IMX250MZR_AoLP.jpg)|

What do the colors in the AoLP image represent? [See the wiki for details](https://github.com/elerac/polanalyser/wiki/How-to-Visualizing-the-AoLP-Image).

### Analysis of Mueller matrix
The [**Mueller matrix**](https://en.wikipedia.org/wiki/Mueller_calculus) represents the change of the polarization state of light. The matrix size is 4x4 (When we consider only linear polarization, the size is 3x3).

We can measure the unknown Mueller matrix by changing the polarization state of both the light and the detector. The following figure shows a schematic diagram to measure the unknown Mueller matrix **M**.
![](documents/mueller_setup.png)
*I* denotes the intensity of the unpolarized light source. **M_PSG** and **M_PSA** represent the Polarization state generator and analyzer (PSG and PSA) in Mueller matrix form. PSG and PSA are commonly composed of the basic optical elements (i.e., linear polarizer and retarder). 
The detector measures the intensity *f* expressed by *f* = [ **M_PSA** **M** **M_PSG** *I* ]00. [...]00 extracts the (0, 0) component of the matrix.

Measuring *f* by changing many combinations of **M_PSG** and **M_PSA** can estimate the unknown Mueller matrix **M** with a linear least-squares method.

The following code shows the example to estimate the 3x3 Mueller matrix image.
```python
import cv2
import polanalyser as pa

# Read all images
folder_path = "dataset/mueller/various3x3"
pimages = pa.io.PolarizationImages(folder_path)

print(len(pimages))  # 16
print(pimages.image[0].shape)  # (2048, 2448)
print(pimages.mueller_psg[0].shape)  # (3, 3)
print(pimages.mueller_psa[0].shape)  # (3, 3)

# Calculate Mueller matrix
img_mueller = pa.calcMueller(pimages.image, 
                             pimages.mueller_psg, 
                             pimages.mueller_psa)

print(img_mueller.shape)  # (2048, 2448, 9)

# Visualize Mueller matrix image
pa.plotMueller("plot_mueller.png", img_mueller, vabsmax=2.0)
```

![](documents/mueller_various.jpg)