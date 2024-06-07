import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union
from pathlib import Path
import re
import numpy as np
import numpy.typing as npt
import cv2
from .json_io import json_read, json_write

PathLike = Union[str, os.PathLike]


def _numerical_sort(value):
    value = str(value)
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def imread(filename_img: PathLike) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """Read a image with properties"""
    filename_img = Path(filename_img)
    filename_json = filename_img.parent / (filename_img.stem + ".json")

    if not filename_img.exists():
        raise FileNotFoundError(f"'{filename_img}' not found.")
    img = cv2.imread(str(filename_img), -1)

    if not filename_json.exists():
        warnings.warn(f"'{filename_json}' not found. Return empty properties.")
        props = dict()
    else:
        props = json_read(filename_json)

    return img, props


def imreadMultiple(filepath: PathLike) -> tuple[npt.NDArray[Any], dict[str, npt.ArrayLike]]:
    """Read multiple images with properties in parallel

    The returned data structure is Structure of arrays (SoA).

    Parameters
    ----------
    filepath : PathLike
        Path to the folder containing image files (.exr, .png) and json files.

    Returns
    -------
    images : np.ndarray
        Images with shape (num_images, height, width, channels)
    props : dict[str, np.ndarray]
        Properties with shape (num_images, ...) for each key. Strutcture of arrays (SoA).
    """
    filepath = Path(filepath)

    # Check if the folder exists
    if not filepath.exists():
        raise FileNotFoundError(f"'{filepath}' not found.")

    if not filepath.is_dir():
        raise FileNotFoundError(f"'{filepath}' is not a folder.")

    # Find image files
    ext_candidates = [".exr", ".png"]
    filenames_img = []
    for ext in ext_candidates:
        filenames_img = sorted(filepath.glob(f"*{ext}"), key=_numerical_sort)
        if len(filenames_img) > 0:
            break
    else:
        raise FileNotFoundError(f"No image files found at '{filepath}'. Supported extensions are {ext_candidates}")

    # Read the first image to determine its shape and type
    # and allocate the output array
    img_0, props_0 = imread(filenames_img[0])
    dtype = img_0.dtype

    num = len(filenames_img)
    multi_img = np.empty((num, *img_0.shape), dtype=dtype)
    multi_props: dict[str, npt.NDArray[Any]] = dict()
    for key in props_0:
        prop = props_0[key]
        if isinstance(prop, str):
            multi_props[key] = np.empty((num,), dtype=object)
        else:
            item_array = np.array(prop)
            multi_props[key] = np.empty((num, *item_array.shape))

    # Set the first image
    multi_img[0] = img_0
    for key in props_0:
        multi_props[key][0] = np.array(props_0[key])

    def imread_and_store(i: int) -> None:
        """Read and store an image with properties"""
        image_i, props_i = imread(filenames_img[i])
        multi_img[i] = image_i.astype(dtype)
        for key in props_i:
            multi_props[key][i] = np.array(props_i[key])

    # Read images in parallel
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(imread_and_store, i) for i in range(1, num)]
        for future in futures:
            future.result()

    return multi_img, multi_props


def imwrite(filename_img: PathLike, image: np.ndarray, **props: dict[str, Any]) -> None:
    """Write a image with properties"""
    filename_img = Path(filename_img)
    filename_img.parent.mkdir(parents=True, exist_ok=True)
    filename_json = filename_img.parent / (filename_img.stem + ".json")
    cv2.imwrite(str(filename_img), image)
    json_write(filename_json, props)


def imwriteMultiple(filepath: PathLike, images: npt.ArrayLike, **props: dict[str, npt.ArrayLike]) -> None:
    """Write multiple images with properties in parallel

    Parameters
    ----------
    filepath : PathLike
        Path to the folder to save images.
    images : npt.ArrayLike
        Images with shape (num_images, height, width, channels)
    props : dict[str, npt.ArrayLike]
        Properties with shape (num_images, ...) for each key. Strutcture of arrays (SoA).
    """
    filepath = Path(filepath)
    if filepath.exists():
        if filepath.is_file():
            raise FileExistsError(f"'{filepath}' is a file, not a folder.")

        # Delete entire folder and files
        for child in filepath.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    else:
        filepath.mkdir(parents=True)

    num_images = len(images)
    dtype = images[0].dtype

    # Convert dtype to np.float32 if np.float64
    if dtype == np.float64:
        images = np.array(images, dtype=np.float32)
        dtype = np.float32

    # Determine the extension
    if dtype == np.float32:
        ext = "exr"
    elif dtype == np.uint8 or dtype == np.uint16:
        ext = "png"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    zfill_width = max(len(str(num_images)), 5)
    list_i_str = [str(i).zfill(zfill_width) for i in range(num_images)]
    filenames_img = [f"image{i_str}.{ext}" for i_str in list_i_str]

    def _imwith(i: int) -> None:
        imwrite(filepath / filenames_img[i], images[i], **{key: prop[i] for key, prop in props.items()})

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_imwith, i) for i in range(num_images)]
        for future in futures:
            future.result()
