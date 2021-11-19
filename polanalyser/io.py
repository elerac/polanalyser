import os
import glob
import re
import json
from typing import Dict, Any, Optional
import numpy as np
import cv2


def numerical_sort(value):
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

class PolarizationImages(list):
    """Provide raw polarization images and paired Mueller matrix (both Polarization State Generator (PSG) and Polarization State Analyzer (PSA)).

    Examples
    --------
    Register and save polarization images with Mueller matix of PSA (e.g., passive measurment using polarization camera)

    >>> pimages = PolarizationImages()
    >>> for _ in range(4):
    ...     image = np.random.rand(256, 256, 3).astype(np.float32)
    ...     mueller_psa = np.random.rand(4, 4)
    ...     pimages.add(image, mueller_psa)
    >>> pimages.save("test_pimages")

    Register and save polarization images with pair of Mueller matrices (e.g., ellipsometer, dual-rotating-retarder)

    >>> pimages = PolarizationImages()
    >>> for _ in range(16):
    ...     image = np.random.rand(256, 256, 3).astype(np.float32)
    ...     mueller_psa = np.random.rand(4, 4)
    ...     mueller_psg = np.random.rand(4, 4)
    ...     pimages.add(image, mueller_psa, mueller_psg)
    >>> pimages.save("test_pimages")

    Load polarization images (iterate)

    >>> pimages = PolarizationImages("test_pimages")
    >>> len(pimages)
    16
    >>> for pimage in pimages:
    ...     image = pimage["image"]
    ...     mueller_psa = pimage["mueller_psa"]
    ...     mueller_psg = pimage["mueller_psg"]

    Load polarization images (list of objects)

    >>> pimages = PolarizationImages("test_pimages")
    >>> images = pimages.image
    >>> type(images)
    <class 'list'>
    >>> len(images)
    16
    >>> type(images[0])
    <class 'numpy.ndarray'>
    >>> images[0].shape
    (256, 256, 3)
    >>> muellers_psg = pimages.mueller_psg
    >>> muellers_psa = pimages.mueller_psa

    Register optional values

    >>> pimages = PolarizationImages()
    >>> image = np.random.rand(256, 256, 3).astype(np.float32)
    >>> mueller_psa = np.random.rand(4, 4)
    >>> pimages.add(image, mueller_psa, polarizer_angle=0)
    >>> pimages.add(image, mueller_psa, polarizer_angle=45)
    >>> pimages.add(image, mueller_psa, polarizer_angle=90)
    >>> pimages.add(image, mueller_psa, polarizer_angle=135)
    >>> pimages.polarizer_angle
    [0, 45, 90, 135]
    """

    def __init__(self, folder_path: Optional[str] = None) -> None:
        super().__init__()

        if folder_path is not None:
            self.load(folder_path)

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            exist_key = False
            values = []
            for item in self:
                val = item.get(name)
                values.append(val)
                if val is not None:
                    exist_key = True
            
            if exist_key:
                return values
            else:
                raise

    def add(self, image: np.ndarray, mueller_psa: np.ndarray, mueller_psg: Optional[np.ndarray] = None, **optional_values: Dict[str, Any]) -> None:
        dict_obj = {"image": image, 
                    "mueller_psa": mueller_psa, 
                    "mueller_psg": mueller_psg, 
                    **optional_values}
        self.append(dict_obj)

    def save(self, folder_path: str, exist_ok: bool = False, ext_img: str = ".exr") -> None:
        os.makedirs(folder_path, exist_ok=exist_ok)

        for i, item in enumerate(self):
            basename_img = f"image{i+1}{ext_img}"
            cv2.imwrite(f"{folder_path}/{basename_img}", item["image"])

            save_dict = {"filename": basename_img, **item}
            del save_dict["image"]

            with open(f"{folder_path}/image{i+1}.json", "w") as f:
                json.dump(save_dict, f, cls=NdarrayEncoder, indent=4)

    def load(self, folder_path: str) -> None:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"'{folder_path}' not found.")

        filenames_json = sorted(glob.glob(f"{folder_path}/*.json"), key=numerical_sort)
        if len(filenames_json) == 0:
            raise FileNotFoundError(f"'.json' files not found at '{folder_path}'")

        self.clear()

        for filename_json in filenames_json:
            with open(filename_json, "r") as f:
                loaded_dict = json.load(f, cls=NdarrayDecoder)
            
            basename_img = loaded_dict.pop("filename")
            filename_img = f"{folder_path}/{basename_img}"
            image = cv2.imread(filename_img, -1)
            if image is None:
                raise FileNotFoundError(f"'{filename_img}' not found.")

            mueller_psa = loaded_dict.pop("mueller_psa")
            mueller_psg = loaded_dict.pop("mueller_psg")

            self.add(image, mueller_psa, mueller_psg, **loaded_dict)