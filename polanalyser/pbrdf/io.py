"""
This code is adapted from the work of [Baek et al., SIGGRAPH 2020].

For more details, visit the original project page:
https://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html

Polanalyser has modified the original code to facilitate its seamless integration into the library.
"""

import struct
from pathlib import Path
from typing import Union, Dict
import numpy as np


def size_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def load(file: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load pBRDF table.

    Parameters
    ----------
    file : Union[str, Path]
        Path to the pBRDF table.

    Returns
    -------
    Dict[str, np.ndarray]
        pBRDF table with auxiliary information.

    Examples
    --------
    >>> filepath_pbsdf = "2_white_billiard_mitsuba/2_white_billiard_inpainted.pbsdf"
    >>> pbrdf_table = pa.pbrdf.load(filepath_pbsdf)
    >>> pbrdf_table.keys()
    dict_keys(['M', 'phi_d', 'theta_d', 'theta_h', 'wvls'])
    >>> pbrdf_table["M"].shape
    (361, 91, 91, 5, 4, 4)
    """

    with open(file, "rb") as f:

        def unpack(fmt):
            result = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
            return result if len(result) > 1 else result[0]

        if f.read(12) != "tensor_file\0".encode("utf8"):
            raise Exception("Invalid tensor file (header not recognized)")

        if unpack("<BB") != (1, 0):
            raise Exception("Invalid tensor file (unrecognized " "file format version)")

        field_count = unpack("<I")

        # Maps from Struct.EType field in Mitsuba
        dtype_map = {1: np.uint8, 2: np.int8, 3: np.uint16, 4: np.int16, 5: np.uint32, 6: np.int32, 7: np.uint64, 8: np.int64, 9: np.float16, 10: np.float32, 11: np.float64}

        fields = {}
        for i in range(field_count):
            field_name = f.read(unpack("<H")).decode("utf8")
            field_ndim = unpack("<H")
            field_dtype = dtype_map[unpack("<B")]
            field_offset = unpack("<Q")
            field_shape = unpack("<" + "Q" * field_ndim)
            fields[field_name] = (field_offset, field_dtype, field_shape)

        result = {}
        for k, v in fields.items():
            f.seek(v[0])
            result[k] = np.fromfile(f, dtype=v[1], count=np.prod(v[2], dtype=np.uint64)).reshape(v[2])

    return result


def save(file: Union[str, Path], **pbrdf_table):
    """Save pBRDF table.

    Parameters
    ----------
    file : Union[str, Path]
        Path to save the pBRDF table.
    pbrdf_table : Dict[str, np.ndarray]
        pBRDF table with auxiliary information.

    Examples
    --------
    >>> M = np.random.rand(361, 91, 91, 5, 4, 4)
    >>> phi_d = np.linspace(0, 2 * np.pi, 361)
    >>> theta_d = np.linspace(0, np.pi, 91)
    >>> theta_h = np.linspace(0, np.pi, 91)
    >>> wvls = np.array([400, 450, 550, 650, 700])
    >>> pbrdf_table = {"M": M, "phi_d": phi_d, "theta_d": theta_d, "theta_h": theta_h, "wvls": wvls}
    >>> pa.pbrdf.save("test.pbsdf", **pbrdf_table)
    """
    align = 8

    with open(file, "wb") as f:
        # Identifier
        f.write("tensor_file\0".encode("utf8"))

        # Version number
        f.write(struct.pack("<BB", 1, 0))

        # Number of fields
        f.write(struct.pack("<I", len(pbrdf_table)))

        # Maps to Struct.EType field in Mitsuba
        dtype_map = {np.uint8: 1, np.int8: 2, np.uint16: 3, np.int16: 4, np.uint32: 5, np.int32: 6, np.uint64: 7, np.int64: 8, np.float16: 9, np.float32: 10, np.float64: 11}

        offsets = {}
        fields = dict(pbrdf_table)

        # Write all fields
        for k, v in fields.items():
            if type(v) is str:
                v = np.frombuffer(v.encode("utf8"), dtype=np.uint8)
            else:
                v = np.ascontiguousarray(v)
            fields[k] = v

            # Field identifier
            label = k.encode("utf8")
            f.write(struct.pack("<H", len(label)))
            f.write(label)

            # Field dimension
            f.write(struct.pack("<H", v.ndim))

            found = False
            for dt in dtype_map.keys():
                if dt == v.dtype:
                    found = True
                    f.write(struct.pack("B", dtype_map[dt]))
                    break
            if not found:
                raise Exception("Unsupported dtype: %s" % str(v.dtype))

            # Field offset (unknown for now)
            offsets[k] = f.tell()
            f.write(struct.pack("<Q", 0))

            # Field sizes
            f.write(struct.pack("<" + ("Q" * v.ndim), *v.shape))

        for k, v in fields.items():
            # Set field offset
            pos = f.tell()

            # Pad to requested alignment
            pos = (pos + align - 1) // align * align

            f.seek(offsets[k])
            f.write(struct.pack("<Q", pos))
            f.seek(pos)

            # Field data
            v.tofile(f)


def main():
    filepath_pbsdf = "pbsdf/2_white_billiard_mitsuba/2_white_billiard_inpainted.pbsdf"
    pbrdf = load_pbsdf(filepath_pbsdf)
    print(pbrdf.keys())
    print(pbrdf["M"].shape)

    M = np.random.rand(361, 91, 91, 5, 4, 4)
    phi_d = np.linspace(0, 2 * np.pi, 361)
    theta_d = np.linspace(0, np.pi, 91)
    theta_h = np.linspace(0, np.pi, 91)
    wvls = np.array([400, 450, 550, 650, 700])
    pbrdf = {"M": M, "phi_d": phi_d, "theta_d": theta_d, "theta_h": theta_h, "wvls": wvls}
    save_pbsdf("test.pbsdf", **pbrdf)


if __name__ == "__main__":
    main()
