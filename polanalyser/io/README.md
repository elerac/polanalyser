# Polarization Image I/O in Polanalyser (experimental)

In general, polarization images require multiple images with some properties (i.e., polarizer angle, Mueller matrix). For example, to acquire a stokes image, we need to capture several images with different polarizer angles. We need an image I/O mechanism to handle several images with associated properties in an easy and fast way. A straightforward approach is to convert these images and properties into a single file format like HDF5 or npy. However, these formats do not allow the images to be viewed in standard image viewers and require additional software to check the images and properties.

The philosophy of Polanalyser is to store the images and properties in a human-readable format while easily accessible by the computer, maintaining the associated images and properties. To achieve this, Polanalyser is designed to store the images and properties in a single folder, where each image is in a standard image format (e.g., exr, png) and the properties are in a json file format. The structure of the folder is as follows:

```
|-- mydata
|   |-- image00000.exr
|   |-- image00000.json
|   |-- image00001.exr
|   |-- image00001.json
|   |-- ...
|   |-- image00015.exr
|   |-- image00015.json
```

The image files are named with a common prefix (e.g., `image`) and a number (e.g., `00000`). The properties are stored in a json file with the same name as the image file. The json file contains the properties in the dictionary format such as:

```json
{
    "mueller_psa": {
        "type": "ndarray",
        "values": [
            [
                0.5,
                0.5,
                0.0
            ],
            [
                0.5,
                0.5,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0
            ]
        ],
        "dtype": "<f8"
    },
    "mueller_psg": {
        "type": "ndarray",
        "values": [
            [
                0.5,
                0.5,
                0.0
            ],
            [
                0.5,
                0.5,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0
            ]
        ],
        "dtype": "<f8"
    }
    "polarizer_angle": 0.0
}
```

To read and write the images and properties, Polanalyser offers `pa.imwriteMultiple` and `pa.imreadMultiple` functions. `pa.imreafdMultiple` reads the images and properties from a single folder and returns the images and properties as numpy arrays and a dictionary, respectively in Structure of Arrays (SoA) format. Both functions are run in parallel enabling fast read and write access. Here's an example of how to use these functions:

```python
import polanalyser as pa

# Read images and properties
images, props = pa.imreadMultiple("mydata")

print(props.keys()) 
# dict_keys(['mueller_psa', 'mueller_psg', 'polarizer_angle'])
print(images.shape) 
# (16, 2048, 2048) 
print(props["mueller_psa"].shape) 
# (16, 3, 3)
print(props["polarizer_angle"].shape) 
# (16,)

# Modify brightness and add new property
brightness_list = []
for i in range(len(images)):
    brightness = i / len(images) 
    images[i] = brightness * images[i]
    brightness_list.append(brightness)
props["brightness"] = brightness_list

# Write new images and properties
pa.imwriteMultiple("mydata_2", images, props)

# Read again
images, props = pa.imreadMultiple("mydata")

print(props.keys()) 
# dict_keys(['mueller_psa', 'mueller_psg', 'polarizer_angle', 'brightness'])
print(props["brightness"])
# [0.  0.0625  0.125  0.1875  0.25  0.3125  0.375  0.4375  0.5  0.5625  0.625  0.6875  0.75  0.8125  0.875  0.9375]
```
