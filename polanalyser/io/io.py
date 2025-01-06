import json
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union, Dict, Tuple, List, Optional
import warnings
import numpy as np
import cv2

PathLike = Union[str, Path]


class PolanalyserWarning(UserWarning):
    pass


class NdarrayEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return {"type": "ndarray", "values": o.tolist(), "dtype": o.dtype.str}
        else:
            return json.JSONEncoder.default(self, o)


def _ndarray_hook(o):
    if "type" in o:
        if o["type"] == "ndarray":
            dtype = o["dtype"] if "dtype" in o else None
            return np.array(o["values"], dtype)
    return o


class NdarrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=_ndarray_hook, *args, **kwargs)


def _save_json(filename_json: PathLike, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
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


def _load_json(filename_json: PathLike) -> Dict[str, Any]:
    """Load dictionary from a json file."""
    filename_json = Path(filename_json)
    with open(filename_json, "r") as f:
        data = json.load(f, cls=NdarrayDecoder)
    return data


def _save_array(name: PathLike, array: np.ndarray) -> Path:
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


def _load_array(filename: PathLike) -> np.ndarray:
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


def _glob_filenames(dirpath: PathLike) -> Tuple[List[Path], List[Path]]:
    """Get filenames of arrays (.png, .exr, .npy) and properties (.json) in a folder."""
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"'{dirpath}' is not a existing folder.")
    suffix_candidates = [".png", ".exr", ".npy"]
    filenames_array = [child for child in dirpath.iterdir() if child.suffix in suffix_candidates]
    filenames_array = sorted(filenames_array, key=_numerical_sort)
    filenames_json = [filename_array.with_suffix(".json") for filename_array in filenames_array]
    return filenames_array, filenames_json


def save(dirpath: PathLike, arrays: Union[np.ndarray, List[np.ndarray]], **kwargs) -> None:
    """Save multiple arrays with properties.

    Parameters
    ----------
    dirpath : PathLike
        The folder path to save arrays and properties.
    arrays : Union[np.ndarray, List[np.ndarray]]
        The list of arrays to save.
    """
    dirpath = Path(dirpath)

    # Create the folder if it does not exist
    if not dirpath.exists():
        dirpath.mkdir(parents=True)

    # Delete the existing files in the folder
    filenames_array, filenames_json = _glob_filenames(dirpath)
    for filename in filenames_array + filenames_json:
        filename.unlink(missing_ok=True)

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
            filename_array = _save_array(_name, _array)
            filename_json = filename_array.with_suffix(".json")
            _save_json(filename_json, _kwargs)
            return filename_array

        z_width = len(str(num))
        furutes = []
        for i in range(len(arrays)):
            name_i = dirpath / f"{i:0{z_width}}"
            arrays_i = arrays[i]
            kwargs_i = {key: kwargs[key][i] for key in kwargs}
            future = executor.submit(task, name_i, arrays_i, **kwargs_i)
            furutes.append(future)

        for future in furutes:
            future.result()


def load(dirpath: PathLike) -> Tuple[List[np.ndarray], Dict[str, List[Any]]]:
    """Load multiple arrays with properties.

    Parameters
    ----------
    dirpath : PathLike
        The folder path containing arrays and properties.

    Returns
    -------
    arrays : List[np.ndarray]
        List of arrays.
    props : Dict[str, List[Any]]
        Dictionary of properties.
    """
    dirpath = Path(dirpath)

    # Check the folder exists
    if not dirpath.is_dir():
        raise FileNotFoundError(f"'{dirpath}' is not a existing folder.")

    filenames_array, filenames_json = _glob_filenames(dirpath)

    # Check the existence of json files
    filenames_missing_json = [str(filename_json) for filename_json in filenames_json if not filename_json.exists()]
    if filenames_missing_json:
        warnings.warn(f"The following json files are missing: {filenames_missing_json}", PolanalyserWarning)

    # Load all json files and get all possible keys
    keys = set()
    for filename_json in filenames_json:
        if filename_json.exists():
            props = _load_json(filename_json)
            keys.update(props.keys())

    # Load arrays and properties in parallel
    with ThreadPoolExecutor() as executor:

        def task(_filename_array: Path, _filename_json: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Load array and properties from a file with json format."""
            array = _load_array(_filename_array)
            if _filename_json.exists():
                props = _load_json(_filename_json)
            else:
                props = {}
            return array, props

        futures = []
        for filename_array, filename_json in zip(filenames_array, filenames_json):
            future = executor.submit(task, filename_array, filename_json)
            futures.append(future)

        arrays = []
        props = {key: [] for key in keys}
        for future in futures:
            array, prop = future.result()
            arrays.append(array)
            for key in keys:
                props[key].append(prop.get(key, None))

    return arrays, props
