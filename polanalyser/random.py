import numpy as np


def _stokes(rng, s0=1.0, dop=None, aolp=None, ellipticity_angle=None, size=None):
    if size == None:
        size = ()

    # Randomly generate values if not provided
    if dop is None:
        dop = rng.uniform(0, 1, size)

    if aolp is None:
        aolp = rng.uniform(0, np.pi, size)

    if ellipticity_angle is None:
        # Apply arcsin to make the distribution uniform
        ellipticity_angle = 0.5 * np.arcsin(1 - 2 * rng.uniform(0, 1, size))

    # Check input values
    if np.any(np.logical_or(dop < 0, 1 < dop)):
        raise ValueError("DoP must be in the range [0, 1]")

    if np.any(np.logical_or(ellipticity_angle < -np.pi / 4, np.pi / 4 < ellipticity_angle)):
        raise ValueError("Ellipticity angle must be in the range [-pi/4, pi/4]")

    if np.any(s0 < 0):
        raise ValueError("Intensity must be non-negative")

    s0 = np.broadcast_to(s0, size)
    s1 = s0 * dop * np.cos(2 * aolp) * np.cos(2 * ellipticity_angle)
    s2 = s0 * dop * np.sin(2 * aolp) * np.cos(2 * ellipticity_angle)
    s3 = s0 * dop * np.sin(2 * ellipticity_angle)
    return np.stack((s0, s1, s2, s3), axis=-1)


def stokes(s0=1.0, dop=None, aolp=None, ellipticity_angle=None, size=None):
    """Randomly generate Stokes vector

    Parameters
    ----------
    s0 : float or array_like, optional
        Intensity of the light. [0, inf). If None, set to 1.
    dop : float or array_like, optional
        Degree of polarization. [0, 1]. If None, generate random DoP.
    aolp : float or array_like, optional
        Angle of linear polarization. [0, pi]. If None, generate random AoLP.
    ellipticity_angle : float or array_like, optional
        Angle of ellipticity. [-pi/4, pi/4]. If None, generate random ellipticity angle.
    size : int or tuple of ints, optional
        Output shape. If None, return a single Stokes vector.

    Returns
    -------
    s : array_like
        Stokes vector. Shape is determined by broadcasting the input arguments.

    Examples
    --------
    >>> s = pa.random.stokes()
    [ 1.         -0.3516055  -0.26569391 -0.63323679]  # Random Stokes vector
    >>> pa.random.stokes(size=(3,)) # (3, 4)
    [[ 1.00000000e+00 -5.91476157e-03 -4.15714440e-04  7.75093548e-02]
     [ 1.00000000e+00  1.47797617e-01 -1.20648980e-01  4.07038091e-02]
     [ 1.00000000e+00 -4.35252249e-02  1.29371584e-02 -9.43606086e-01]] # Random Stokes vectors (array)

    Fix DoP

    >>> s = pa.random.stokes(dop=0.5, size=(5))
    >>> pa.cvtStokesToDoP(s)
    [0.5 0.5 0.5 0.5 0.5]
    >>> pa.cvtStokesToAoLP(s)
    [0.06065768 0.60803243 2.43605145 2.49862427 0.8271829 ]  # Random AoLP
    >>> pa.cvtStokesToEllipticityAngle(s)
    [ 0.22203514  0.55317095  0.52020158 -0.62225768  0.48007445]  # Random ellipticity angle

    Fix DoP and AoLP

    >>> s = pa.random.stokes(dop=0.5, aolp=0.1, size=(5))
    >>> pa.cvtStokesToDoP(s)
    [0.5 0.5 0.5 0.5 0.5]
    >>> pa.cvtStokesToAoLP(s)
    [0.1 0.1 0.1 0.1 0.1]
    >>> pa.cvtStokesToEllipticityAngle(s)
    [-0.0908445  -0.62325593 -0.14761242 -0.65347546  0.76911759]  # Random ellipticity angle
    """
    return _stokes(np.random, s0, dop, aolp, ellipticity_angle, size)


class PolarizationGenerator(np.random.Generator):
    def __init__(self, bit_generator: np.random.BitGenerator):
        super().__init__(bit_generator)

    def stokes(self, s0=1.0, dop=None, aolp=None, ellipticity_angle=None, size=None):
        return _stokes(self, s0, dop, aolp, ellipticity_angle, size)


def default_rng(seed=None) -> PolarizationGenerator:
    """Create a new PolarizationGenerator with default BitGenerator (PCG64).

    Examples
    --------
    >>> rng = pa.random.default_rng()
    >>> rng.stokes()
    [ 1.         -0.3516055  -0.26569391 -0.63323679]  # Random Stokes vector
    """
    return PolarizationGenerator(np.random.PCG64(seed))
