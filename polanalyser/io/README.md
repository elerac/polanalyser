# Data I/O in Polanalyser (experimental)

Polarization imaging often involves capturing multiple images with associated properties, such as polarizer angles or Mueller matrices. For example, acquiring Stokes parameters requires capturing several images at different polarizer rotation angles. A straightforward approach is to store the images and their associated metadata in a single file format, such as HDF5 or NPZ. While these formats offer flexibility in storing these images and metadata, they are incompatible with standard image viewers and require specialized software to access the data. This limitation hinders efficient debugging and verification of data during acquisition and processing. To overcome these challenges, Polanalyser introduces a data format that integrates images with their metadata in a human-readable and machine-parsable structure. Additionally, Polanalyser provides I/O functions enable efficient parallelized saving and loading of data.

## Data Format

### Folder Structure

Polanalyser organizes data within a single folder to ensure simplicity and accessibility. Each image is stored in a standard image format (PNG, EXR) or a numpy array format (NPY), while its associated metadata is saved in a corresponding JSON file. The image file and its metadata file share the same stem name, ensuring easy pairing. The folder structure is as follows:

```shell
|-- mydata
|   |-- 00.png
|   |-- 00.json
|   |-- 01.png
|   |-- 01.json
|   |-- ...
|   |-- 15.png
|   |-- 15.json
```

### JSON File

The JSON file contains the properties of the image in a structured format. The properties are stored as a dictionary, where the keys represent the property names and the values are the property values. Below is an example of a JSON file containing the angles and Mueller matrix of an analyzer:

```json
{
    "angles": 1.2566370614359172,
    "mm": {
        "type": "ndarray",
        "values": [
            [
                0.5,
                -0.40450849718747367,
                0.2938926261462366,
                0.0
            ],
            [
                -0.40450849718747367,
                0.3272542485937368,
                -0.2377641290737884,
                0.0
            ],
            [
                0.2938926261462366,
                -0.2377641290737884,
                0.17274575140626322,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0
            ]
        ],
        "dtype": "<f8"
    }
}
```

## I/O Functions

Polanalyser provides the `pa.save` and `pa.load` functions for saving and loading images and their properties. Both functions are parallelized for multiple image saving and loading for efficient processing.

### `pa.save(filepath, arrays, **kwargs)`

This function writes images and their properties to a specified folder. The numbering of images and properties corresponds to the order of the input list. The file format is determined by the shape and data type of the input arrays.

#### Parameters

- `filepath` (PathLike): The path to the folder where the images and properties are saved.
- `arrays` (List[np.ndarray]): A list of images to be saved.
- `**kwargs`: Properties to be saved. Lists of values (e.g., int, float, np.ndarray).

### `pa.load(filepath)`

This function reads images and their properties from a specified folder.

#### Parameters

- `filepath` (PathLike): The path to the folder where the images and properties are saved.

#### Returns

- `images` (Union[np.ndarray, List[np.ndarray]]): A list of images loaded from the folder.
- `props` (Dict[str, List[Any]]): A dictionary of properties loaded from the folder.

### Example Usage

```python
import numpy as np
import polanalyser as pa

# Save images and properties 
images = [np.random.uniform(0, 255, (400, 600, 3)).astype(np.uint8) for _ in range(16)]
angles = np.linspace(0, np.pi, 16)
mm = [pa.polarizer(ang) for ang in angles]
pa.save("mydata", images, angles=angles, mm=mm)

# Load images and properties
images, props = pa.load("mydata") # List of images, dictionary of properties
angles = props["angles"] # List of float
mm = props["mm"] # List of 2D arrays
```
