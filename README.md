<p align="center">
    <img alt="polanalyser logo" src="documents/polanalyser_logo.png" height="90em">
</p>

---

Polanalyser is polarization image analysis tool.

## Key Features

- [Demosaicing of a image captured by a polarization image sensor](#polarization-demosaicing)
  - Both Monochrome/Color Polarization image sensors (*e.g.*, IMX250MZR / MYR) are supported.
- [Analysis of Stokes vector](#analysis-of-stokes-vector)
  - Calculate Stokes vector from an image of a polarization camera.
  - Convert Stokes vector to meaningful parameters, such as Degree of Linear Polarization (DoLP), Angle of Linear Polarization (AoLP).
- [Analysis of Mueller matrix](#analysis-of-mueller-matrix)
  - Provide basic Mueller matrix elements, such as polarizer, retarder, and rotator.
  - Calculate Mueller matrix from images captured under a variety of polarimetric conditions.
- [Visualizing polarimetric images](#visualizing-polarimetric-images)
  - Apply colormap to polarization images, such as AoLP and ToP.
  - Plot Mueller matrix image.
- [Symbolic Stokes-Mueller computation](#symbolic-stokes-mueller-computation)
  - Calculate the Stokes vector and Mueller matrix symbolically to understand complex combinations of optical elements.

## Polarization Image Dataset

Dataset of images captured by a polarization camera (FLIR, BFS-U3-51S5P-C) is available.

[**[Click here to download the dataset (Google Drive)]**](https://drive.google.com/drive/folders/1vCe9N05to5_McvwyDqxTmLIKz7vRzmbX?usp=sharing)

## Dependencies and Installation

- Numpy
- OpenCV
- matplotlib
- SymPy (optional)

```sh
pip install polanalyser
```

## Usage

### Polarization demosaicing

Demosaic  raw polarization image captured with a polarization sensor (*e.g.*, [IMX250MZR / MYR](https://www.sony-semicon.com/en/products/is/industry/polarization.html)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="documents/demosaicing_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="documents/demosaicing_light.png">
  <img alt="demosaicing" src="documents/demosaicing_light.png">
</picture>

```python
import cv2
import polanalyser as pa

img_raw = cv2.imread("dataset/dragon.png", 0)

img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw, pa.COLOR_PolarMono)
```

### Analysis of Stokes vector

[**Stokes vector**](https://en.wikipedia.org/wiki/Stokes_parameters) describes the polarization states. We can measure these values by using a *linear polarizer* (To measure the circular polarization $s_3$, we also need to use a *retarder*).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="documents/stokes_setup_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="documents/stokes_setup_light.png">
  <img alt="Stokes setup" src="documents/stokes_setup_light.png">
</picture>

Stokes vector can be converted to meaningful values. *Degree of Linear Polarization* (DoLP) represents how much the light is polarized. The value is 1 for perfectly polarized light and 0 for unpolarized light. *Angle of Linear Polarization* (AoLP) represents the polarization angle of the incident light relative to the camera sensor axis. The value ranges from 0 to 180 degrees.

```python
import cv2
import numpy as np
import polanalyser as pa

# Read image and demosaicing
img_raw = cv2.imread("dataset/dragon.png", 0)
img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw, pa.COLOR_PolarMono)

# Calculate the Stokes vector per-pixel
image_list = [img_000, img_045, img_090, img_135]
angles = np.deg2rad([0, 45, 90, 135])
img_stokes = pa.calcStokes(image_list, angles)

# Decompose the Stokes vector into its components
img_s0, img_s1, img_s2 = cv2.split(img_stokes)

# Convert the Stokes vector to Intensity, DoLP and AoLP
img_intensity = pa.cvtStokesToIntensity(img_stokes)
img_dolp = pa.cvtStokesToDoLP(img_stokes)
img_aolp = pa.cvtStokesToAoLP(img_stokes)
```

||Example of results | |
|:-:|:-:|:-:|
|Intensity ($s_0$)|DoLP|AoLP|
|![](documents/dragon_IMX250MZR_intensity.jpg)|![](documents/dragon_IMX250MZR_DoLP.jpg)|![](documents/dragon_IMX250MZR_AoLP.jpg)|

### Analysis of Mueller matrix

[**Mueller matrix**](https://en.wikipedia.org/wiki/Mueller_calculus) represents the change of the polarization state of light. The matrix size is 4x4 (When we consider only linear polarization, the size is 3x3).

Polanalyzer provides basic Mueller matrix elements in numpy array. For example, the following code shows the Mueller matrix of a linear polarizer and a quarter-wave plate.

```python
import polanalyser as pa

# Linear polarizer
M_LP = pa.polarizer(0)
print(M_LP)
# [[0.5 0.5 0.  0. ]
#  [0.5 0.5 0.  0. ]
#  [0.  0.  0.  0. ]
#  [0.  0.  0.  0. ]]

# Quarter-wave plate
M_QWP = pa.qwp(0)
print(M_QWP)
# [[ 1.  0.  0.  0.]
#  [ 0.  1.  0.  0.]
#  [ 0.  0.  0.  1.]
#  [ 0.  0. -1.  0.]]
```

We can measure the unknown Mueller matrix by changing the polarization state of both the light and the detector. The following figure shows a schematic diagram to measure the unknown Mueller matrix $\mathbf{M}$

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="documents/mueller_setup_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="documents/mueller_setup_light.png">
  <img alt="Mueller setup" src="documents/mueller_setup_light.png">
</picture>

$I$ denotes the intensity of the unpolarized light source. $\mathbf{M}_\textrm{PSG}$ and $\mathbf{M}_\textrm{PSA}$ represent the Polarization state generator and analyzer (PSG and PSA) in Mueller matrix form. PSG and PSA are commonly composed of the basic optical elements (i.e., linear polarizer and retarder).
The detector measures the intensity $f$ expressed by $f = [ \mathbf{M}_\textrm{PSA} \mathbf{M} \mathbf{M}_\textrm{PSG} I]_{00}$. $[...]_{00}$ extracts the (0, 0) component of the matrix.

Measuring $f$ by changing many combinations of $\mathbf{M}_\textrm{PSG}$ and $\mathbf{M}_\textrm{PSA}$ can estimate the unknown Mueller matrix $\mathbf{M}$ with a linear least-squares method.

The following code shows the example to estimate the 3x3 Mueller matrix image.

```python
import cv2
import polanalyser as pa

# Read all images
path = "dataset/toy_example_3x3_pc"
pcontainer = pa.PolarizationContainer(path)
image_list = pcontainer.get_list("image")
mueller_psg_list = pcontainer.get_list("mueller_psg")
mueller_psa_list = pcontainer.get_list("mueller_psa")

print(len(pcontainer))  # 16
print(image_list[0].shape)  # (2048, 2448)
print(mueller_psg_list[0].shape)  # (3, 3)
print(mueller_psa_list[0].shape)  # (3, 3)

# Calculate Mueller matrix
img_mueller = pa.calcMueller(image_list, mueller_psg_list, mueller_psa_list)

print(img_mueller.shape)  # (2048, 2448, 3, 3)

# Visualize Mueller matrix image
pa.plotMueller("plot_mueller.png", img_mueller, vabsmax=2.0)
```

![](documents/mueller_various.jpg)


### Visualizing polarimetric images

### Stokes vector visualization

Polanalyser provides functions to visualize Stokes vector images, such as AoLP, DoLP, ToP (Type of Polarization), and CoP (Chirality of Polarization). The color mapping is designed based on the relevant papers [[Wilkie and Weidlich, SCCG2010]](https://dl.acm.org/doi/10.1145/1925059.1925070), [[Baek+, SIGGRAPH2020]](http://vclab.kaist.ac.kr/siggraph2020/index.html), [[Jeon+, CVPR2024]](https://eschoi.com/SPDataset/). Note that this mapping may slightly differ from the original papers.

```python
# Example of visualization functions
img_aolp_vis = pa.applyColorMapToAoLP(img_aolp)
img_dop_vis = pa.applyColorMapToDoLP(img_dolp)
img_top_vis = pa.applyColorToToP(img_ellipticity_angle, img_dop)
img_cop_vis = pa.applyColorToCoP(img_ellipticity_angle)
```

Here is an example of visualizing the Stokes vector images. The image is from [spectro-polarimetric dataset](https://huggingface.co/datasets/jyj7913/spectro-polarimetric).

|||||
|:-:|:-:|:-:|:-:|
| $s_0$ | $s_1$ | $s_2$ | $s_3$ |
|![](documents/visualization/color.jpeg)|![](documents/visualization/s1.jpeg)|![](documents/visualization/s2.jpeg)|![](documents/visualization/s3.jpeg)|
| DoLP | AoLP | AoLP (light) | AoLP (dark) |
|![](documents/visualization/dolp.jpeg)|![](documents/visualization/aolp.jpeg)|![](documents/visualization/aolp_light.jpeg)|![](documents/visualization/aolp_dark.jpeg)|
| DoP | DoCP | ToP | CoP |
|![](documents/visualization/dop.jpeg)|![](documents/visualization/docp.jpeg)|![](documents/visualization/top.jpeg)|![](documents/visualization/cop.jpeg)|

In AoLP visualization, Polanalyser provides three types of AoLP visualization: AoLP, AoLP (light), and AoLP (dark). For more details, [see the wiki page](https://github.com/elerac/polanalyser/wiki/How-to-Visualizing-the-AoLP-Image).


### Symbolic Stokes-Mueller computation

This feature supports the symbolic computation of the Stokes vector and Mueller matrix powered by SymPy. This feature is particularly useful for understanding the effects of complex combinations of optical elements. 

Here are examples of Malus's law and an ellipsometer. We can symbolically obtain the intensity of light passing through the sequence of optical elements without tedious calculations by hand.

```python
from sympy import symbols, simplify
import polanalyser.sympy as pas

theta = symbols("theta", real=True)

# Example 1: Malus's law
M_L1 = pas.polarizer(0)
M_L2 = pas.polarizer(theta)
f = (M_L2 @ M_L1)[0, 0]  
print(simplify(f))  
# -> 0.5*cos(theta)**2

# Example 2: Ellipsometer [Azzam+, 1978][Baek+, 2020]
M = pas.mueller()  # Symbolic Mueller matrix
I = 1.0
M_PSG = pas.qwp(theta) @ pas.polarizer(0)
M_PSA = pas.polarizer(0) @ pas.qwp(5 * theta)
f = (M_PSA @ M @ M_PSG * I)[0, 0] 
print(simplify(f))  
# -> 0.25*m00 + 0.25*m10*cos(10*theta)**2 + 0.125*m20*sin(20*theta) - 0.25*m30*sin(10*theta) + 0.25*(m01 + m11*cos(10*theta)**2 + m21*sin(20*theta)/2 - m31*sin(10*theta))*cos(2*theta)**2 + 0.25*(m02 + m12*cos(10*theta)**2 + m22*sin(20*theta)/2 - m32*sin(10*theta))*sin(2*theta)*cos(2*theta) + 0.25*(m03 + m13*cos(10*theta)**2 + m23*sin(20*theta)/2 - m33*sin(10*theta))*sin(2*theta)
```

