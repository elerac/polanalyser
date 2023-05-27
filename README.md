<p align="center">
    <img alt="polanalyser logo" src="documents/polanalyser_logo.png" height="90em">
</p>

---

Polanalyser is polarization image analysis tool.

## Key Features

- [Demosaicing of a image captured by a polarization image sensor](#polarization-demosaicing)
  - Both Monochrome/Color Polarization image sensors (*e.g.*, IMX250MZR / MYR) are supported
  - Three algorithms are implemented; Bilinear, Variable Number of Gradients (VNG), Edge-Aware (EA)
- [Analysis of Stokes vector](#analysis-of-stokes-vector)
  - Calculate Stokes vector from an image of a polarization camera
  - Convert Stokes vector to meaningful parameters, such as Degree of Linear Polarization (DoLP), Angle of Linear Polarization (AoLP)
- [Analysis of Mueller matrix](#analysis-of-mueller-matrix)
  - Calculate Mueller matrix from images captured under a variety of polarimetric conditions
  - Provide basic Mueller matrix
- Visualizing a polarimetric image
  - Apply colormap to AoLP image
  - Plot Mueller matrix image

## Polarization Image Dataset

Dataset of images captured by a polarization camera (FLIR, BFS-U3-51S5P-C) is available.

[**[Click here to download the dataset (Google Drive)]**](https://drive.google.com/drive/folders/1vCe9N05to5_McvwyDqxTmLIKz7vRzmbX?usp=sharing)

## Requirement and Installation

- Numpy
- OpenCV
- matplotlib

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

[**Stokes vector**](https://en.wikipedia.org/wiki/Stokes_parameters) describes the polarization states. We can measure these values by using a *linear polarizer* (To measure the circular polarization S3, we also need to use a *retarder*).

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
|Intensity (S0)|DoLP|AoLP|
|![](documents/dragon_IMX250MZR_intensity.jpg)|![](documents/dragon_IMX250MZR_DoLP.jpg)|![](documents/dragon_IMX250MZR_AoLP.jpg)|

What do the colors in the AoLP image represent? [See the wiki for details](https://github.com/elerac/polanalyser/wiki/How-to-Visualizing-the-AoLP-Image).

### Analysis of Mueller matrix

[**Mueller matrix**](https://en.wikipedia.org/wiki/Mueller_calculus) represents the change of the polarization state of light. The matrix size is 4x4 (When we consider only linear polarization, the size is 3x3).

We can measure the unknown Mueller matrix by changing the polarization state of both the light and the detector. The following figure shows a schematic diagram to measure the unknown Mueller matrix **M**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="documents/mueller_setup_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="documents/mueller_setup_light.png">
  <img alt="Mueller setup" src="documents/mueller_setup_light.png">
</picture>

*I* denotes the intensity of the unpolarized light source. **M_PSG** and **M_PSA** represent the Polarization state generator and analyzer (PSG and PSA) in Mueller matrix form. PSG and PSA are commonly composed of the basic optical elements (i.e., linear polarizer and retarder).
The detector measures the intensity *f* expressed by *f* = [ **M_PSA** **M** **M_PSG** *I* ]00. [...]00 extracts the (0, 0) component of the matrix.

Measuring *f* by changing many combinations of **M_PSG** and **M_PSA** can estimate the unknown Mueller matrix **M** with a linear least-squares method.

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

Below is a method to calibrate your polarimetry instrument using premeasured reference samples. This calibration method comes from the research paper at: https://doi.org/10.1364/AO.46.008533 The calibration reference samples you use should span the area or volume of the Poincare sphere you wish to analyze. This particular calibration method makes no assumptions about the polarimetry instrument, it simply maps a relationship of how 
the instrument's results correspond to the reference sample. In addition, this method attempts to correct for imperfections in alignment, the polarization filter efficiency, camera noise, etc by using only the most 
significant single values from the single value decomposition used to calculate the pseudo inverse matrix to produce the calibration matrix W. 

Some notes before you attempt to calibrate your polarimetry instrument:

The Mueller matrix is esquisitely sensitive to changes on the surface of a material, this is very similar to the technique used to measure nm thick films in the semiconductor industry. So be sure to not touch the surface and store in a dry, dust-free container. If the samples must be cleaned then remeasure their Mueller matrix afterwards.

The optical constants depend not only on wavelength but also temperature. Be sure your reference samples are at the same temperature as they were when measured.

The measurements will be wavelength dependent and the narrower the wavelength of light used will correspond to a more precise and accurate result because multiple wavelengths aren't being measured at once. I suggest using at least LEDs for a narrow band peak. A separate W calibration matrix will be required for each wavelength if your instrument is multispectral.

The way the reference samples are measured matters. If you plan to build a transmission style polarimetry instrument the reference samples will need to be measured on a transmission style polarimetry instrument, likewise for reflective instruments. 

The following code shows how to produce and save a calibration matrix for calculating Mueller matrices. 

```python
import polanalyser as pa

# Read all images
path = "dataset/cal_example_3x3_pc"
pcontainer = pa.PolarizationContainer(path)
image_list = pcontainer.get_list("image")
mueller_psr_list = pcontainer.get_list("mueller_psr")

print(len(pcontainer))  # N*M
print(image_list[0].shape)  # (2048, 2448)
print(mueller_psr_list[0].shape)  # (4, 4)

# Calculate W matrix
W = pa.calcW(image_list, mueller_psr_list)

# Save W matrix to a CSV file named WCal.csv
ps.saveW(W)
```

The following code shows how to load and use the calibration matrix to produce a Meuller matrix image, as shown above without calibration.

```python
import polanalyser as pa

# Read all images
path = "dataset/cal_example_3x3_pc"
pcontainer = pa.PolarizationContainer(path)
image_list = pcontainer.get_list("image")

print(len(pcontainer))  # 36  6 filter configurations for PSG and 6 for PSA, 9 for 3 filter configurations
print(image_list[0].shape)  # (2048, 2448)

# Load calibration matrix
W = loadW()

# Use the calibration matrix to calculate the Mueller matrix from the intensity images
M = calcM(image_list, W, (4,4)) # shape at the end is either (3,3) or (4,4) depending if you used just linear filters or not

# Plot the matrix as before
pa.plotMueller("plot_mueller.png", M, vabsmax=2.0)
```
To use this calibration method, you will first need to make a list of the configurations you wish to use to calculate the Mueller matrices for your samples. The number of measurements per sample, N, can be calculated from: N = # configurations of PSA * # configurations of PSG. Check that the number of measurements equals the expected number of measurements. Order the list of measurements in a regular fashion so that the measurements can be taken in the same order for each sample. 

My measurement naming and ordering system:

0 - 0 degree polarization 

4 - 45 degree polarization

9 - 90 degree polarization

3 - 135 degree polarization

r - right circular polarization

l - left circular polarization

image[PSG filter][PSA filter][sample #]

example:

image041.json - image[0 degree polarization][45 degree polarization][sample 1]

I will order them in ascending order by angle and alphabetically, l first.

image000.json

image040.json

image090.json

image0l0.json

image0r0.json

...

imager03.json

imager43.json

imager93.json

imagerl3.json

imagerr4.json

The only word required is "image" for the image names, everything else is optional. The sample number is not included except for when generating the W calibration matrix using reference samples. Otherwise, the sample number is simply a name or number to keep track of the sample being measured. The only time multiple samples will be measured in the same run is when producing the W calibration matrix. 

Preparing the json configuration files for the calibration:

The json files are prepared similarly to the files that are used in the toy example for calculating the Mueller matrix. For example:

```json
{
	"filename": "image043.exr",
	"mueller_ref": {
		"type": "ndarray",
		"values": [
		[
			0.5,
			0.5,
			0.0,
			0.0
		],
		[
			0.5,
			0.5,
			0.0,
			0.0
		],
		[
			0.0,
			0.0,
			0.0,
			0.0
		],
		[
			0.0,
			0.0,
			0.0,
			0.0
		]
	]
	}
}
```

The 4x4 matrix here is formatted just as the 3x3 matrix is in the toy example json. To produce the calibration json files you will replace the 4x4 matrix values with the Mueller matrix values you have obtained by having the Mueller matrix of your reference samples measured on an already calibrated and validated instrument. The Mueller matrix you enter for a given sample will be the same, only the filename will change to reflect which image the configuration file goes with. The json and image files will have the same name except for the file type. This could easily be automated with a short script.

The polarization filters should cover the entirety of the imaging field, otherwise the image will need to be cropped down to a rectangle of just the reference polarization filter. If cropped, the images must all be cropped to the same size, and the samples measured on the instrument will need to be cropped down as well. This calibration method calibrates each pixel of the camera. 

The intensity image for a given calibration configuration file is simply the image obtained with the instrument in the correct configuration and the sample in place. Once all the intensity images and corresponding configuration files have been measured and made, you're ready to make your calibration matrix.

To use your new W calibration matrix simply follow the steps above for saving it to a file then loading that file to determine the Mueller matrix of a sample. A calibration matrix also makes the Meuller matrix calculation much faster because most of the linear algebra is performed when calculating the W matrix.
