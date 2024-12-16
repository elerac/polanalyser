import json
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union, Dict, Tuple, List, Optional
import numpy as np
import cv2

PathLike = Union[str, Path]


class NdarrayEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return {"type": "ndarray", "values": o.tolist(), "dtype": o.dtype.str}
        else:
            return json.JSONEncoder.default(self, o)


def ndarray_hook(o):
    if "type" in o:
        if o["type"] == "ndarray":
            dtype = o["dtype"] if "dtype" in o else None
            return np.array(o["values"], dtype)
    return o


class NdarrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=ndarray_hook, *args, **kwargs)


def save_json(filename_json: PathLike, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Save dictionary to a json file."""
    filename_json = Path(filename_json)

    if data is None:
        data = {}

    # Merge data and kwargs
    data = {**data, **kwargs}

    # Convert numpy array with one element into a standard Python scalar object.
    for key in data:
        item = data[key]
        if isinstance(item, (np.ndarray, np.generic)):
            if item.size == 1:
                data[key] = item.item()

    with open(filename_json, "w") as f:
        json.dump(data, f, cls=NdarrayEncoder, indent=4)


def load_json(filename_json: PathLike) -> Dict[str, Any]:
    """Load dictionary from a json file."""
    filename_json = Path(filename_json)
    with open(filename_json, "r") as f:
        data = json.load(f, cls=NdarrayDecoder)
    return data


def save_array(name: PathLike, array: np.ndarray) -> Path:
    """Save array to a file in uncompressed format.

    The file format is determined by the array shape and dtype.
    Supported formats are:
    - png (uint8, uint16 image)
    - exr (float32 image)
    - npy (any array)
    """
    name = Path(name)
    suffix = name.suffix
    if suffix != "":  # if the suffix is specified
        filename = name
        filename.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".npy":
            np.save(filename, array)
            return filename
        else:
            cv2.imwrite(str(filename), array)
            return filename
    else:  # if the suffix is not specified
        shape = array.shape
        dtype = array.dtype
        ndim = array.ndim

        # png (uint8, uint16 image)
        if (ndim == 2 or (ndim == 3 and shape[-1] in [3, 4])) and (dtype in [np.uint8, np.uint16]):
            suffix = ".png"
            filename = Path(name).with_suffix(suffix)
            filename.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(filename), array, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            return filename

        # exr (float32 image)
        if (ndim == 2 or (ndim == 3 and shape[-1] == 3)) and (dtype == np.float32):
            suffix = ".exr"
            filename = Path(name).with_suffix(suffix)
            filename.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(filename), array)
            return filename

        # npy (any array)
        suffix = ".npy"
        filename = Path(name).with_suffix(suffix)
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, array)
        return filename


def load_array(filename: PathLike) -> np.ndarray:
    """Load array from a file.

    It supports image files (e.g., png, exr) and npy files.
    """
    filename = Path(filename)
    suffix = filename.suffix

    if not filename.exists():
        raise FileNotFoundError(f"'{filename}' does not exist.")

    if suffix == ".npy":
        return np.load(filename)

    return cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)


# ==================================================================================================


def _numerical_sort(value):
    """Sort the file names numerically."""
    value = str(value)
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def _get_filenames(filepath: PathLike) -> Tuple[List[Path], List[Path]]:
    """Get filenames of arrays (.png, .exr, .npy) and properties (.json) in a folder."""
    filepath = Path(filepath)
    suffix_candidates = [".png", ".exr", ".npy"]
    filenames_array = [child for child in filepath.iterdir() if child.suffix in suffix_candidates]
    filenames_array = sorted(filenames_array, key=_numerical_sort)
    filenames_json = [filename_array.with_suffix(".json") for filename_array in filenames_array]
    return filenames_array, filenames_json


def save(filepath: PathLike, arrays: Union[np.ndarray, List[np.ndarray]], **kwargs):
    """Save multiple arrays with properties."""
    filepath = Path(filepath)

    # Create the folder if it does not exist
    if not filepath.exists():
        filepath.mkdir(parents=True)

    # Delete the all files in the folder
    filenames_array, filenames_json = _get_filenames(filepath)
    for filename in filenames_array + filenames_json:
        filename.unlink()

    # Check the size of properties is consistent with the number of arrays
    num = len(arrays)
    for key in kwargs:
        num_key = len(kwargs[key])
        if num_key != num:
            raise ValueError(f"The size of '{key}' does not match the number of arrays. The expected length is {num}, but got {num_key}.")

    # Save arrays and properties in parallel
    with ThreadPoolExecutor() as executor:

        def task(_name: PathLike, _array: np.ndarray, **_kwargs) -> Path:
            """Save array to a file with properties in json format."""
            filename_array = save_array(_name, _array)
            filename_json = filename_array.with_suffix(".json")
            save_json(filename_json, _kwargs)
            return filename_array

        z_width = len(str(num))
        furutes = []
        for i in range(len(arrays)):
            name_i = filepath / f"{i:0{z_width}}"
            arrays_i = arrays[i]
            kwargs_i = {key: kwargs[key][i] for key in kwargs}
            future = executor.submit(task, name_i, arrays_i, **kwargs_i)
            furutes.append(future)

        for future in furutes:
            future.result()


def load(filepath: PathLike) -> Tuple[List[np.ndarray], Dict[str, List[Any]]]:
    """Load multiple arrays with properties."""
    filepath = Path(filepath)

    # Check the folder exists
    if not filepath.is_dir():
        raise FileNotFoundError(f"'{filepath}' is not a existing folder.")

    filenames_array, filenames_json = _get_filenames(filepath)

    # Load arrays and properties in parallel
    with ThreadPoolExecutor() as executor:

        def task(_filename_array: PathLike, _filename_json: PathLike) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Load array and properties from a file with json format."""
            array = load_array(_filename_array)
            props = load_json(_filename_json)
            return array, props

        futures = []
        for filename_array, filename_json in zip(filenames_array, filenames_json):
            future = executor.submit(task, filename_array, filename_json)
            futures.append(future)

        arrays = []
        props = {}
        for future in futures:
            array, prop = future.result()
            arrays.append(array)
            for key in prop:
                if key not in props:
                    props[key] = []
                props[key].append(prop[key])

    return arrays, props
