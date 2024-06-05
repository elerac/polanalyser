import os
import json
from pathlib import Path
from typing import Any, Union
import numpy as np

PathLike = Union[str, os.PathLike]


class NdarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"type": "ndarray", "values": obj.tolist(), "dtype": obj.dtype.str}
        else:
            return json.JSONEncoder.default(self, obj)


class NdarrayDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "type" in obj:
            if obj["type"] == "ndarray":
                dtype = obj["dtype"] if "dtype" in obj else None
                return np.array(obj["values"], dtype)

        return obj


def json_write(filename_json: PathLike, data: dict[str, Any]) -> None:
    filename_json = Path(filename_json)

    # Convert numpy array with one element into a standard Python scalar object.
    for key in data:
        item = data[key]
        if isinstance(item, (np.ndarray, np.generic)):
            if item.size == 1:
                data[key] = item.item()

    with open(filename_json, "w") as f:
        json.dump(data, f, cls=NdarrayEncoder, indent=4)


def json_read(filename_json: PathLike) -> dict[str, Any]:
    filename_json = Path(filename_json)
    with open(filename_json, "r") as f:
        data = json.load(f, cls=NdarrayDecoder)
    return data
