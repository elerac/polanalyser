# Polanalyser
Polanalyser is polarization image analysis tool.

It can be used for 
* [Demosaicing of bayer images taken with a polarization camera](#polarization-demosaicing)
* [Analysis of Stokes vector](#analysis-of-stokes-vector)
* [Analysis of Mueller matrix](#analysis-of-mueller-matrix)

By analyzing the polarization images, it can be used for [various applications](https://github.com/elerac/polanalyser/wiki/Applications-for-Polarization-Image).

### Note
Currently, **only linear polarization** is assumed, and circular polarization is not taken into account.

## Requirement
* OpenCV
* Numpy
* matplotlib
* Numba

## Installation
```sh
pip install git+https://github.com/elerac/polanalyser
```

## Polarization Image Dataset
A dataset of images taken by a polarization camera (FLIR, BFS-U3-51S5P-C) is available.

[**Click here and download dataset**](https://drive.google.com/drive/folders/1vCe9N05to5_McvwyDqxTmLIKz7vRzmbX?usp=sharing)


## Usage
### Polarization demosaicing
Demosaic raw polarization image taken with the polarization sensor (e.g. [IMX250MZR](https://www.sony-semicon.co.jp/e/products/IS/polarization/product.html)).
![](documents/demosaicing.png)
```python
import cv2
import polanalyser as pa

img_raw = cv2.imread("dataset/dragon.png", 0)

img_demosaiced = pa.demosaicing(img_raw)

img_0, img_45, img_90, img_135 = cv2.split(img_demosaiced)
```

### Analysis of Stokes vector
The [**Stokes vector**](https://en.wikipedia.org/wiki/Stokes_parameters) (or parameters) are a set of values that describe the polarization state. You can get these values by taking at least three images while rotating the polarizer (If you want to take into account circular polarization, you need to add measurements with a retarder).
![](documents/stokes_setup.png)
The Stokes vector can be converted to meaningful values. **Degree of Linear Polarization** (DoLP) represents how much the light is polarized. The value is 1 for perfectly polarized light and 0 for unpolarized light. **Angle of Linear Polarization** (AoLP) represents the polarization angle of the incident light relative to the camera sensor axis. The value ranges from 0 to 180 degrees.
```python
import cv2
import numpy as np
import polanalyser as pa

# Read image and demosaicing
img_raw = cv2.imread("dataset/dragon.png", 0)
img_demosaiced = pa.demosaicing(img_raw)

# Calculate the Stokes vector per-pixel
radians = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
img_stokes = pa.calcStokes(img_demosaiced, radians)

# Decompose the Stokes vector into its components
img_S0, img_S1, img_S2 = cv2.split(img_stokes)

# Convert the Stokes vector to Intensity, DoLP and AoLP
img_intensity = pa.cvtStokesToIntensity(img_stokes)
img_DoLP      = pa.cvtStokesToDoLP(img_stokes)
img_AoLP      = pa.cvtStokesToAoLP(img_stokes)
```

||Example of results | |
|:-:|:-:|:-:|
|Intensity (S0/2.0)|DoLP|AoLP|
|![](documents/dragon_IMX250MZR_intensity.jpg)|![](documents/dragon_IMX250MZR_DoLP.jpg)|![](documents/dragon_IMX250MZR_AoLP.jpg)|

What do the colors in the AoLP image represent? [See the wiki for details](https://github.com/elerac/polanalyser/wiki/How-to-Visualizing-the-AoLP-Image).

### Analysis of Mueller matrix
The [**Mueller matrix**](https://en.wikipedia.org/wiki/Mueller_calculus) is a 4x4 matrix that represents the change in the polarization state of light. If we consider only linear polarization, it can be represented as a 3x3 matrix.
When a light changes its polarization state due to optical elements or reflection, the changed polarization state can be computed by the matrix product of the Muller matrix and the Stokes vector.
![](documents/mueller_setup.png)
For linear polarization, the Mueller matrix can be obtained by placing linear polarizers on the light side and the camera side. It is calculated using the least-squares method from multiple images taken by rotating each polarizer (9 or more images).
```python
import cv2
import numpy as np
import polanalyser as pa

# Read all images (l:light, c:camera)
img_l0_c0     = cv2.imread("dataset/mueller/various_l0_c0.exr", -1)
img_l0_c45    = cv2.imread("dataset/mueller/various_l0_c45.exr", -1)
...
img_l135_c135 = cv2.imread("dataset/mueller/various_l135_c135.exr", -1)

# Prepare variables to be put into the function
images         = cv2.merge([img_l0_c0, img_l0_c45, ... , img_l135_c135])
radinas_light  = np.array([0, 0, ..., np.pi*3/4])
radians_camera = np.array([0, np.pi/4, ..., np.pi*3/4])

# Calculate the Muller matrix per-pixel
img_mueller = pa.calcMueller(images, radians_light, radians_camera)

# Decompose the Mueller matrix into its components
img_m11, img_m12, img_m13,\
img_m21, img_m22, img_m23,\
img_m31, img_m32, img_m33  = cv2.split(img_mueller)

# Plot the Mueller matrix image
pa.plotMueller("plot_mueller.jpg", img_mueller, vabsmax=0.5)
```

Here's an example of the result.
![](documents/mueller_various.jpg)