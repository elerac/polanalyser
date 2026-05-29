import numpy as np
import numpy.typing as npt

from .stokes import stokes as stokes_from_params


def _random_stokes(rng, s0=None, dop=None, aolp=None, eang=None, size=None):
    """Shared implementation for random.stokes and PolarizationGenerator.stokes."""
    if size is None:
        size = ()

    if s0 is None:
        s0 = rng.uniform(0, 1, size)

    if dop is None:
        dop = rng.uniform(0, 1, size)

    if aolp is None:
        aolp = rng.uniform(0, np.pi, size)

    if eang is None:
        # The arcsin makes the distribution uniform
        eang = 0.5 * np.arcsin(1 - 2 * rng.uniform(0, 1, size))

    return stokes_from_params(s0=s0, dop=dop, aolp=aolp, eang=eang)


def stokes(
    s0: npt.ArrayLike | None = 1.0,
    dop: npt.ArrayLike | None = None,
    aolp: npt.ArrayLike | None = None,
    eang: npt.ArrayLike | None = None,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Randomly generate Stokes vector.

    Parameters
    ----------
    s0 : float or array_like, optional
        Intensity of the light, in [0, inf). Defaults to 1.0.
        If None, generate random intensity [0, 1).
    dop : float or array_like, optional
        Degree of polarization. [0, 1]. If None, generate random DoP.
    aolp : float or array_like, optional
        Angle of linear polarization. [0, pi]. If None, generate random AoLP.
    eang : float or array_like, optional
        Ellipticity angle. [-pi/4, pi/4]. If None, generate random ellipticity angle.
    size : int or tuple of ints, optional
        Output shape. If None, return a single Stokes vector.

    Returns
    -------
    stokes : ndarray, ``(*size, 4)``
        Stokes vector. Shape is determined by broadcasting the input arguments.

    Examples
    --------
    Generate a single random Stokes vector.

    >>> pa.random.stokes()
    [ 1.         -0.3516055  -0.26569391 -0.63323679]

    Generate multiple random Stokes vectors.

    >>> pa.random.stokes(size=3) # (3, 4)
    [[ 1.          0.10098732  0.0643837   0.38024539]
     [ 1.          0.85911994  0.4263879  -0.23704069]
     [ 1.          0.09153383 -0.11594401 -0.41531663]]
    >>> pa.random.stokes(size=(512, 1024))  # (512, 1024, 4)

    Generate random Stokes vectors with specified DoP=0.5.

    >>> s = pa.random.stokes(dop=0.5, size=5)
    >>> pa.stokes_to_dop(s)
    [0.5 0.5 0.5 0.5 0.5]
    >>> pa.stokes_to_aolp(s)
    [0.06065768 0.60803243 2.43605145 2.49862427 0.8271829 ]
    >>> pa.stokes_to_eang(s)
    [ 0.22203514  0.55317095  0.52020158 -0.62225768  0.48007445]

    Generate random Stokes vectors with specified DoP=0.5 and AoLP=0.1.

    >>> s = pa.random.stokes(dop=0.5, aolp=0.1, size=5)
    >>> pa.stokes_to_dop(s)
    [0.5 0.5 0.5 0.5 0.5]
    >>> pa.stokes_to_aolp(s)
    [0.1 0.1 0.1 0.1 0.1]
    >>> pa.stokes_to_eang(s)
    [-0.0908445  -0.62325593 -0.14761242 -0.65347546  0.76911759]
    """
    return _random_stokes(np.random, s0, dop, aolp, eang, size)


class PolarizationGenerator(np.random.Generator):
    def __init__(self, bit_generator: np.random.BitGenerator):
        super().__init__(bit_generator)

    def stokes(self, s0=1.0, dop=None, aolp=None, eang=None, size=None):
        return _random_stokes(self, s0, dop, aolp, eang, size)


def default_rng(seed=None) -> PolarizationGenerator:
    """Create a new PolarizationGenerator with default BitGenerator (PCG64).

    Examples
    --------
    >>> rng = pa.random.default_rng()
    >>> rng.stokes()
    [ 1.         -0.3516055  -0.26569391 -0.63323679]  # Random Stokes vector
    """
    return PolarizationGenerator(np.random.PCG64(seed))
