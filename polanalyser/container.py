import os
import glob
import re
import json
from collections.abc import Container
from typing import List, Dict, Any, Optional, Union, SupportsIndex
import numpy as np
import cv2


def _numerical_sort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class NdarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"type": "ndarray", "values": obj.tolist()}
        else:
            return json.JSONEncoder.default(self, obj)


class NdarrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "type" in obj:
            if obj["type"] == "ndarray":
                return np.array(obj["values"])

        return obj


class PolarizationContainer(Container):
    """Specialized container datatype for polarization image and paired Mueller matrix (both Polarization State Generator (PSG) and Polarization State Analyzer (PSA)).

    Examples
    --------
    Register and save polarization images with Mueller matix of PSA (e.g., passive measurment using polarization camera)

    >>> pcontainer = pa.PolarizationContainer()
    >>> for deg in [0, 45, 90, 135]:
    ...     image = np.random.rand(256, 256, 3).astype(np.float32)
    ...     mueller_psa = pa.polarizer(np.deg2rad(deg))
    ...     pcontainer.append(image, mueller_psa)
    >>> pcontainer.save("test_pcontainer")

    Register and save polarization images with pair of Mueller matrices (e.g., ellipsometer, dual-rotating-retarder)

    >>> pcontainer = pa.PolarizationContainer()
    >>> for _ in range(16):
    ...     image = np.random.rand(256, 256, 3).astype(np.float32)
    ...     mueller_psa = np.random.rand(4, 4)
    ...     mueller_psg = np.random.rand(4, 4)
    ...     pcontainer.append(image, mueller_psa, mueller_psg)
    >>> pcontainer.save("test_pcontainer")

    Save ".npy" format

    >>> pcontainer.save("test_pcontainer_npy", ext_img=".npy")

    Load polarization images (iterate)

    >>> pcontainer = pa.PolarizationContainer("test_pcontainer")
    >>> len(pcontainer)
    16
    >>> for pdata in pcontainer:
    ...     image = pdata["image"]
    ...     mueller_psa = pdata["mueller_psa"]
    ...     mueller_psg = pdata["mueller_psg"]

    Load polarization images (list of objects)

    >>> pcontainer = pa.PolarizationContainer("test_pcontainer")
    >>> image_list = pcontainer.get_list("image")
    >>> type(image_list)
    <class 'list'>
    >>> len(image_list)
    16
    >>> type(image_list[0])
    <class 'numpy.ndarray'>
    >>> image_list[0].shape
    (256, 256, 3)
    >>> mueller_psg_list = pcontainer.get_list("mueller_psg")
    >>> mueller_psa_list = pcontainer.get_list("mueller_psa")

    Register optional values

    >>> pcontainer = pa.PolarizationContainer()
    >>> image = np.random.rand(256, 256, 3).astype(np.float32)
    >>> mueller_psa = np.random.rand(4, 4)
    >>> pcontainer.append(image, mueller_psa, polarizer_angle=0)
    >>> pcontainer.append(image, mueller_psa, polarizer_angle=45)
    >>> pcontainer.append(image, mueller_psa, polarizer_angle=90)
    >>> pcontainer.append(image, mueller_psa, polarizer_angle=135)
    >>> pcontainer.get_list("polarizer_angle")
    [0, 45, 90, 135]
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.data = list()  # List of dict

        if path is not None:
            self.load(path)

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, i: Union[SupportsIndex, slice]):
        return self.data.__getitem__(i)

    def __contains__(self, o: object) -> bool:
        return self.data.__contains__(o)

    def append(self, image: np.ndarray, mueller_psa: Optional[np.ndarray] = None, mueller_psg: Optional[np.ndarray] = None, **opt_values: Dict[str, Any]) -> None:
        """Add the pair of an image and Mueller matrix(matrices)

        Parameters
        ----------
        image : np.ndarray
            Image
        mueller_psa : Optional[np.ndarray], optional
            Mueller matrix of PSG, by default None
        mueller_psg : Optional[np.ndarray], optional
            Mueller matrix of PSA, by default None
        """
        pdata = {"image": image, "mueller_psa": mueller_psa, "mueller_psg": mueller_psg, **opt_values}
        self.data.append(pdata)

    def get_list(self, key: Any) -> List[Any]:
        """Returns the values in list format for the specified key

        Parameters
        ----------
        key : Any
            Key to access

        Returns
        -------
        values : List[Any]
            The values in list
        """
        exist_key = False
        values = []
        for item in self.data:
            val = item.get(key)
            values.append(val)
            if val is not None:
                exist_key = True

        if not exist_key:
            raise KeyError(f"{key}")

        return values

    def save(self, path: str, exist_ok: bool = False, ext_img: str = ".exr") -> None:
        """Save images and its properties

        Parameters
        ----------
        path : str
            Path to save
        exist_ok : bool, optional
            _description_, by default False
        ext_img : str, optional
            Extension of image, by default ".exr".
            The following extensions are available.
            - General image extension supported by OpenCV (e.g., ".png", ".jpg", ".exr")
            - NumPy's NPY format. (i.e., ".npy")
        """
        os.makedirs(path, exist_ok=exist_ok)

        zfill_width = len(str(len(self.data)))
        for i, item in enumerate(self.data):
            i_str = str(i + 1).zfill(zfill_width)

            # Export image
            basename_img = f"image{i_str}{ext_img}"
            path_img = f"{path}/{basename_img}"
            image = item["image"]
            if ext_img == ".npy":
                np.save(path_img, image)
            else:
                cv2.imwrite(path_img, image)

            # Export json
            # (export filename of an image instead of actual values)
            save_dict = {"filename": basename_img, **item}
            del save_dict["image"]
            with open(f"{path}/image{i_str}.json", "w") as f:
                json.dump(save_dict, f, cls=NdarrayEncoder, indent=4)

    def load(self, path: str) -> None:
        """Load images and its properties

        Parameters
        ----------
        path : str
            Path to load
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"'{path}' not found.")

        filenames_json = sorted(glob.glob(f"{path}/*.json"), key=_numerical_sort)
        if len(filenames_json) == 0:
            raise FileNotFoundError(f"'.json' files not found at '{path}'")

        self.data.clear()

        for filename_json in filenames_json:
            # Load json file
            with open(filename_json, "r") as f:
                loaded_dict = json.load(f, cls=NdarrayDecoder)

            # Read image
            basename_img = loaded_dict.pop("filename")
            path_img = f"{path}/{basename_img}"
            if not os.path.exists(path_img):
                raise FileNotFoundError(f"'{path_img}' not found.")

            _, ext_img = os.path.splitext(basename_img)
            if ext_img == ".npy":
                image = np.load(path_img)
            else:
                image = cv2.imread(path_img, -1)

            # Mueller matrix
            mueller_psa = loaded_dict.pop("mueller_psa", None)
            mueller_psg = loaded_dict.pop("mueller_psg", None)

            self.append(image, mueller_psa, mueller_psg, **loaded_dict)
