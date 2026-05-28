from typing import Sequence, overload

import numpy as np
import numpy.typing as npt

from . import random, stokes


def _broadcast_mueller(entries: Sequence[Sequence[npt.ArrayLike]]) -> np.ndarray:
    """Build a broadcasted Mueller matrix from a readable 4x4 grid of entries.

    This helper supports a convenient way to write a Mueller matrix as a nested
    4x4 container whose entries may be scalars or array-like objects. All 16
    entries are broadcast to a common batch shape and then arranged into a single
    numeric array with trailing Mueller-matrix axes. This keeps the physical
    matrix layout visible in constructors such as polarizers, retarders, and
    depolarizers, while still allowing batched inputs such as image-shaped
    parameters.

    Unlike ``np.array(entries)``, which may fail or produce an object array when
    entries have different but broadcast-compatible shapes, this function applies
    NumPy broadcasting entry-wise before assembling the final array.

    Parameters
    ----------
    entries : Sequence[Sequence[npt.ArrayLike]]
        A 4x4 nested sequence of broadcast-compatible Mueller matrix entries.
        Each entry may be a scalar or an array-like object with batch dimensions.

    Returns
    -------
    mueller : ndarray, shape (..., 4, 4)
        Numeric Mueller matrix array. The leading axes are the broadcasted batch
        shape, and the last two axes index the Mueller matrix rows and columns.

    Examples
    --------
    Build a Mueller matrix with a common image batch shape ``(H, W)`` while keeping
    the readable 4x4 matrix layout.

    >>> H, W = 100, 200
    >>> d1 = np.random.rand(H, W)
    >>> d2 = np.random.rand(H, W)
    >>> d3 = np.random.rand(H, W)
    >>> M = __broadcast_mueller(
    ...     [
    ...         [1.0, 0.0, 0.0, 0.0],
    ...         [0.0,  d1, 0.0, 0.0],
    ...         [0.0, 0.0,  d2, 0.0],
    ...         [0.0, 0.0, 0.0,  d3],
    ...     ]
    ... )
    >>> M.shape
    (100, 200, 4, 4)
    """
    rows = [list(row) for row in entries]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("Expected a 4x4 nested sequence of Mueller entries.")

    flat = [np.asarray(x) for row in rows for x in row]
    b = np.broadcast_arrays(*flat)
    grid = np.array(b, dtype=np.result_type(*b))
    grid = grid.reshape(4, 4, *b[0].shape)
    return np.moveaxis(grid, (0, 1), (-2, -1))


def estimate_mueller(intensities: npt.ArrayLike, mm_psg: npt.ArrayLike, mm_psa: npt.ArrayLike) -> np.ndarray:
    """Estimate Mueller matrix from measured intensities and Mueller matrices of Polarization State Generator (PSG) and Polarization State Analyzer (PSA)

    This function calculates Mueller matrix image from intensity images captured under a variety of polarimetric conditions (both PSG and PSA).
    Polarimetric conditions are specified by the Mueller matrices (`mm_psg` and `mm_psa`).

    The unknown Mueller matrix is calculated by the least-squares method from pairs of intensities and Muller matrices.
    The number of input pairs must be greater than the number of Mueller matrix parameters (i.e., more than 9 or 16).

    Parameters
    ----------
    intensities : ArrayLike
        Intensities (N, *), where N is the number of intensities and * is the shape of the image (or tensor).
    mm_psg : ArrayLike
        Mueller matrix of the Polarization State Generator (PSG) in (N, 3, 3) or (N, 4, 4). Stokes vector is also available in (N, 3) or (N, 4).
    mm_psa : ArrayLike
        Mueller matrix of the Polarization State Analyzer (PSA) in (N, 3, 3) or (N, 4, 4). Stokes vector is also available in (N, 3) or (N, 4).

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix. The last two channels are correspondence to Mueller matrix. (*, 3, 3) or (*, 4, 4)

    Examples
    --------
    >>> mueller_obj = 2 * np.random.rand(4, 4) - 1  # Unknown mueller matrix of target object
    >>> # mueller_obj = 2 * np.random.rand(128, 256, 4, 4) - 1  # You can expand to array (like image)
    >>> intensity_list = []
    >>> mm_psg_list = []
    >>> mm_psa_list = []
    >>> for angle in np.linspace(0, np.pi, num=36, endpoint=False):
    ...     mm_psg = pa.qwp(5 * angle) @ pa.polarizer(0)
    ...     mm_psa = pa.polarizer(np.pi / 2) @ pa.qwp(angle)
    ...     intensity = (mm_psa @ mueller_obj @ mm_psg)[..., 0, 0]
    ...     intensity_list.append(intensity)
    ...     mm_psg_list.append(mueller_psg)
    ...     mm_psa_list.append(mueller_psa)
    >>> mueller_pred = pa.calcMueller(intensity_list, mm_psg_list, mm_psa_list)
    >>> mueller_pred.shape
    (4, 4)
    >>> np.allclose(mueller_obj, mueller_pred)
    True
    """
    # Convert ArrayLike object to np.ndarray
    intensities = np.array(intensities)  # (N, *)
    mm_psa = np.array(mm_psa)  # (N, 3, 3) or (N, 4, 4) or (N, 3) or (N, 4)
    mm_psg = np.array(mm_psg)  # (N, 3, 3) or (N, 4, 4) or (N, 3) or (N, 4)
    num = len(intensities)

    # Check the number of the input elements
    if not (len(intensities) == len(mm_psa) == len(mm_psg)):
        raise ValueError(f"The number of elements must be same. {len(intensities)} != {len(mm_psa)} != {len(mm_psg)}")

    # Check the shape of the Mueller matrices
    if not (mm_psg.ndim in (2, 3) and mm_psa.ndim in (2, 3)):
        raise ValueError(f"The shape of mueller matrices must be (N, 3, 3) or (N, 4, 4) or (N, 3) or (N, 4), not {mm_psg.shape} or {mm_psa.shape}")

    # In case of stokes vector (N, 3) or (N, 4), expand the axis for later matrix manipulation
    if mm_psg.ndim == 2:  # (N, 3) or (N, 4) -> (N, 3, 1) or (N, 4, 1)
        mm_psg = mm_psg[:, :, np.newaxis]

    if mm_psa.ndim == 2:  # (N, 3) or (N, 4) -> (N, 1, 3) or (N, 1, 4)
        mm_psa = mm_psa[:, np.newaxis, :]

    m_h = mm_psg.shape[1]
    m_w = mm_psa.shape[2]

    # Construct the observation matrix
    W = np.empty((num, m_h * m_w))
    for i in range(num):
        s_psg = mm_psg[i, :, 0][:, np.newaxis]  # [m00, m10, m20] or [m00, m10, m20, m30], (3, 1) or (4, 1)
        s_psa = mm_psa[i, 0, :][np.newaxis, :]  # [m00, m01, m02] or [m00, m01, m02, m03], (1, 3) or (1, 4)
        W[i] = np.ravel((s_psg @ s_psa).T)
    W_pinv = np.linalg.pinv(W)

    intensities = np.moveaxis(intensities, 0, -1)  # (*, N)

    # Least-squares
    mueller = np.tensordot(W_pinv, intensities, axes=(-1, -1))  # (9, *) or (16, *)
    mueller = np.moveaxis(mueller, 0, -1)  # (*, 9) or (*, 16)
    mueller = np.reshape(mueller, (*mueller.shape[:-1], m_h, m_w))  # (*, 3, 3) or (*, 4, 4)
    return mueller


def _normalize(vec, norm=None):
    """Normalize a vector to unit length"""
    # vec: (..., 3)
    if norm is None:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    if norm.ndim == vec.ndim - 1:
        norm = norm[..., None]
    return np.divide(vec, norm, out=np.zeros_like(vec), where=(norm != 0))


def retardance_vector(M_R: npt.ArrayLike) -> np.ndarray:
    """Convert retardance Mueller matrix to retardance vector

    The equations are based on the paper by Lu and Chipman (1996).

    Parameters
    ----------
    M_R : array-like (..., 4, 4)
        Retardance Mueller matrix. If the matrix is not a retardance matrix, the result would be incorrect.

    Returns
    -------
    R_vec: array (..., 3)
        Retardance vector (..., 3)
    """
    M_R = np.asarray(M_R)
    M_R = np.asarray(M_R, dtype=np.promote_types(np.float32, M_R.dtype))
    if M_R.shape[-2:] != (4, 4):
        raise ValueError(f"Invalid shape: {M_R.shape}. Expected (..., 4, 4).")

    R = np.arccos(np.clip(np.trace(M_R, axis1=-2, axis2=-1) / 2 - 1, -1.0, 1.0))  # Eq. (17)
    levi_civita_ijk = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        ]
    )
    m_R = M_R[..., 1:, 1:]  # Eq. (14)
    R_hat = 1 / (2 * np.sin(R[..., None])) * np.sum(levi_civita_ijk * m_R[..., None, :, :], axis=(-2, -1))  # Eq. (17)
    R_vec = R[..., None] * R_hat  # Eq. (8)
    return R_vec



def lu_chipman_decompose(M: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose Mueller matrix with Lu-Chipman decomposition method.

    Lu-Chipman decomposition method (1996) [1]_ decomposes a Mueller matrix :math:`\\mathbf{M}`
    into depolarization :math:`\\mathbf{M}_\\Delta`, retardance :math:`\\mathbf{M}_R` , and diattenuation :math:`\\mathbf{M}_D` components as

    .. math::
        \\mathbf{M} = \\mathbf{M}_\\Delta \\cdot \\mathbf{M}_R \\cdot \\mathbf{M}_D.

    Parameters
    ----------
    M : array_like (..., 4, 4)
        Mueller matrix.

    Returns
    -------
    M_Delta : np.ndarray (..., 4, 4)
        Depolarization Mueller matrix.
    M_R : np.ndarray (..., 4, 4)
        Retardance Mueller matrix.
    M_D : np.ndarray (..., 4, 4)
        Diattenuation Mueller matrix.

    References
    ----------
    .. [1] Shih-Yau Lu and Russell A Chipman. Interpretation of Mueller matrices based on polar decomposition. Journal of the Optical Society of America A (JOSA A) 13, 5 (1996), 1106-1113.

    Examples
    --------
    Compose a Mueller matrix.

    >>> M_Delta = pa.depolarizer([0.9, 0.8, 0.7])
    >>> M_R = pa.retarder(np.deg2rad(20), np.deg2rad(30))
    >>> M_D = pa.diattenuator([0.3, 0.2, 0.1])
    >>> M = M_Delta @ M_R @ M_D  # (4, 4)

    Decompose the Mueller matrix with Lu-Chipman method.

    >>> M_Delta_dec, M_R_dec, M_D_dec = pa.lu_chipman_decompose(M)

    Verify the decomposition.

    >>> np.allclose(M_Delta, M_Delta_dec)
    True
    >>> np.allclose(M_R, M_R_dec)
    True
    >>> np.allclose(M_D, M_D_dec)
    True
    """
    # References of the variable names
    # --------------------------------
    # M_Delta: Depolarizer Mueller matrix (Eq.(46))
    # M_R: Retardance Mueller matrix (Eq.(14))
    # M_D: Diattenuation Mueller matrix (Eq.(18))
    # D: Diattenuation (Eq.(1))
    # D_vec: Diattenuation vector (Eq.(2))
    # R: Retardance (Eq.(10))
    # R_vec: Retardance vector (Eq.(8))
    # P: Polarizance (Eq.(31))
    # P_vec: Polarizance vector (Eq.(32))
    # m_Delta: 3x3 submatrix of M_Delta (Eq.(52))
    # m_R: 3x3 submatrix of M_R (Eq.(15))
    # m_D: 3x3 submatrix of M_D (Eq.(19))

    M = np.asarray(M)
    if M.shape[-2:] != (4, 4):
        raise ValueError(f"Invalid shape: {M.shape}. Expected (..., 4, 4).")

    dtype = M.dtype
    I3 = np.eye(3, dtype=dtype)
    shape = M.shape[:-2]

    # Flatten batch
    M_flat = M.reshape(-1, 4, 4)  # (N, 4, 4)
    N = M_flat.shape[0]
    m00 = M_flat[:, 0, 0]  # (N,)

    # Diattenuation (Eqs. (1),(2),(18),(19),(28))
    D_vec = 1 / m00[:, None] * M_flat[:, 0, 1:]
    D = np.linalg.norm(D_vec, axis=-1)
    D = np.clip(D, 0.0, 1.0)  # Avoid numerical errors
    D_hat = _normalize(D_vec, norm=D)

    sqrt1_D2 = np.sqrt(1.0 - D**2)[:, None, None]
    D_hat_outer = np.einsum("ni,nj->nij", D_hat, D_hat)
    m_D = sqrt1_D2 * I3 + (1.0 - sqrt1_D2) * D_hat_outer

    M_D = np.empty((N, 4, 4), dtype=dtype)
    M_D[:, 0, 0] = 1
    M_D[:, 0, 1:] = D_vec
    M_D[:, 1:, 0] = D_vec
    M_D[:, 1:, 1:] = m_D
    M_D *= m00[:, None, None]

    # Polarizance (Eqs. (31),(32))
    P_vec = 1 / m00[:, None] * M_flat[:, 1:, 0]
    P = np.linalg.norm(P_vec, axis=-1)
    P_hat = _normalize(P_vec, norm=P)

    # Check singularity of M_D (D == 1)
    singular = np.isclose(D, 1.0)
    nonsingular = ~singular

    M_Delta = np.empty((N, 4, 4), dtype=dtype)
    M_R = np.empty((N, 4, 4), dtype=dtype)

    # Non-singular branch (Section 7)
    if nonsingular.any():
        idx = nonsingular
        N_nonsingular = np.sum(idx)
        M_ = M_flat[idx]
        M_D_ = M_D[idx]
        m00_ = m00[idx]
        D_vec_ = D_vec[idx]
        D_ = D[idx]
        P_vec_ = P_vec[idx]

        # M' = M @ inv(M_D) (Eq. 47)
        M_prime_ = np.linalg.solve(np.swapaxes(M_D_, -1, -2), np.swapaxes(M_, -1, -2))
        M_prime_ = np.swapaxes(M_prime_, -1, -2)

        # 3x3 submatrix of M
        m_ = M_[:, 1:, 1:] / m00_[:, None, None]

        # Polarizance vector of the depolarizer (Eq. 50)
        P_Delta_vec_ = (P_vec_ - np.einsum("nij,nj->ni", m_, D_vec_)) / (1.0 - D_**2)[:, None]

        # 3x3 submatrix of M' (Eqs. 48, 51)
        m_prime_ = M_prime_[:, 1:, 1:]

        # m'(m'T)
        m_prime_m_prime_T_ = np.einsum("nij,nkj->nik", m_prime_, m_prime_)

        # Eigenvalues of m'(m'T)
        lam = np.linalg.eigvalsh(m_prime_m_prime_T_)  # Use eigvalsh for real symmetric PSD
        lam = np.clip(lam, 0.0, None)  # Avoid numerical errors
        l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]

        # 3x3 submatrix of M_Delta (Eq. 52)
        s12 = np.sqrt(l1 * l2)
        s23 = np.sqrt(l2 * l3)
        s31 = np.sqrt(l3 * l1)
        s_sum = np.sqrt(l1) + np.sqrt(l2) + np.sqrt(l3)
        s_prod = np.sqrt(l1 * l2 * l3)
        A = m_prime_m_prime_T_ + (s12 + s23 + s31)[:, None, None] * I3
        B = (s_sum)[:, None, None] * m_prime_m_prime_T_ + (s_prod)[:, None, None] * I3
        m_Delta_ = np.linalg.solve(A, B)  # inv(A) @ B
        m_Delta_ *= np.sign(np.linalg.det(m_Delta_))[:, None, None]

        # M_Delta (Eq. 48)
        M_Delta_ = np.zeros((N_nonsingular, 4, 4), dtype=dtype)
        M_Delta_[:, 0, 0] = 1
        M_Delta_[:, 0, 1:] = 0
        M_Delta_[:, 1:, 0] = P_Delta_vec_
        M_Delta_[:, 1:, 1:] = m_Delta_

        # M_R (Eq. 53)
        M_R_ = np.linalg.solve(M_Delta_, M_prime_)  # inv(M_Delta) @ M'

        M_Delta[idx] = M_Delta_
        M_R[idx] = M_R_

    # Singular branch (Appendix A)
    if singular.any():
        idx = singular
        N_singular = np.sum(idx)
        P_ = P[idx]
        P_hat_ = P_hat[idx]
        D_vec_ = D_vec[idx]

        # M_Delta for singular case (Eq. A3)
        M_Delta_ = np.zeros((N_singular, 4, 4), dtype=dtype)
        M_Delta_[:, 0, 0] = 1
        M_Delta_[:, 1:, 1:] = P_[:, None, None] * I3

        # M_R for singular case (Eq. A4)
        cos_arg = np.einsum("ni,ni->n", P_hat_, D_vec_).clip(-1.0, 1.0)
        R_axis = np.cross(P_hat_, D_vec_)
        R_vec = _normalize(R_axis) * np.arccos(cos_arg)[:, None]
        M_R_ = retardance_matrix(R_vec)  # Eqs. 14, 15

        M_Delta[idx] = M_Delta_
        M_R[idx] = M_R_

    # Restore original batch shape
    M_Delta = M_Delta.reshape(*shape, 4, 4)
    M_R = M_R.reshape(*shape, 4, 4)
    M_D = M_D.reshape(*shape, 4, 4)

    return M_Delta, M_R, M_D


def rotator(theta: npt.ArrayLike) -> np.ndarray:
    """Mueller matrix of the rotator

    Parameters
    ----------
    theta : array-like, (...,)
        The angle of rotation

    Returns
    -------
    mueller : np.ndarray, (..., 4, 4)
        Mueller matrix.
    """
    theta = np.asarray(theta)
    s = np.sin(2.0 * theta)
    c = np.cos(2.0 * theta)
    return _broadcast_mueller(
        [
            [1, 0, 0, 0],
            [0, c, s, 0],
            [0, -s, c, 0],
            [0, 0, 0, 1],
        ]
    )


def rotateMueller(mueller: npt.ArrayLike, theta: npt.ArrayLike) -> np.ndarray:
    """Rotate Mueller matrix

    Parameters
    ----------
    mueller : array-like, (..., 3, 3) or (..., 4, 4)
        Mueller matrix to rotate.
    theta : array-like, (...,)
        The angle of rotation

    Returns
    -------
    mueller_rotated : np.ndarray
        Rotated mueller matrix (..., 3, 3) or (..., 4, 4)
    """
    mueller = np.asarray(mueller)
    theta = np.asarray(theta)
    n_rows, n_cols = mueller.shape[-2:]
    if n_rows != n_cols or n_rows > 4:
        raise ValueError(f"The shape of mueller must be (..., 3, 3) or (..., 4, 4), not {mueller.shape}.")
    return rotator(-theta)[..., :n_rows, :n_cols] @ mueller @ rotator(theta)[..., :n_rows, :n_cols]


def polarizer(theta: npt.ArrayLike) -> np.ndarray:
    """Mueller matrix of the linear polarizer.

    Parameters
    ----------
    theta : array_like, (...,)
        Rotation angle of the linear polarizer.

    Returns
    -------
    mueller : np.ndarray, (..., 4, 4)
        Mueller matrix of the linear polarizer.

    Examples
    --------
    >>> pa.polarizer(np.deg2rad(0))
    [[0.5, 0.5, 0.0, 0.0],
     [0.5, 0.5, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0]]
    >>> pa.polarizer(np.deg2rad(45))
    [[0.5, 0.0, 0.5, 0.0],
     [0.0, 0.0, 0.0, 0.0],
     [0.5, 0.0, 0.5, 0.0],
     [0.0, 0.0, 0.0, 0.0]]
    """
    theta = np.asarray(theta)
    s = np.sin(2.0 * theta)
    c = np.cos(2.0 * theta)
    return 0.5 * _broadcast_mueller(
        [
            [1, c, s, 0],
            [c, c * c, s * c, 0],
            [s, s * c, s * s, 0],
            [0, 0, 0, 0],
        ]
    )


@overload
def retarder(delta: npt.ArrayLike, theta: npt.ArrayLike) -> np.ndarray: ...
@overload
def retarder(R_vec: npt.ArrayLike) -> np.ndarray: ...


def retarder(arg0, arg1=None) -> np.ndarray:
    """Mueller matrix of a retarder

    Supports two call signatures:
    - `retarder(delta, theta)`: linear retarder with phase delay `delta` [rad] and fast-axis angle `theta` [rad].
    - `retarder(R_vec)`: general retarder specified by a retardance vector `R_vec` (..., 3).

    Parameters
    ----------
    delta or R_vec : float or array-like, shape (..., 3) when array-like
        If float, the phase difference between the fast and slow axis delta in radians.
        If array-like, the retardance vector R_vec; the last axis must have length 3.
    theta : float, optional
        Fast-axis angle theta in radians.

    Returns
    -------
    np.ndarray
        Mueller matrix (4, 4)
    """
    if arg1 is not None:
        # retarder(delta, theta)
        delta = arg0
        theta = arg1
        s = np.sin(delta)
        c = np.cos(delta)
        mueller = _broadcast_mueller(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, c, s],
                [0, 0, -s, c],
            ]
        )
        mueller = rotateMueller(mueller, theta)
        return mueller
    else:
        arg0 = np.asarray(arg0)
        if arg0.shape[-1] != 3:
            raise ValueError(f"Invalid shape: {arg0.shape}. Expected (..., 3).")

        # retarder(R_vec)
        R_vec = arg0
        return retardance_matrix(R_vec)


def qwp(theta: npt.ArrayLike) -> np.ndarray:
    """Mueller matrix of Quarter-Wave Plate (QWP).

    Parameters
    ----------
    theta : array_like, (...,)
        Angle of the fast axis.

    Returns
    -------
    mueller : np.ndarray, (..., 4, 4)
        Mueller matrix.

    Examples
    --------
    >>> pa.qwp(np.deg2rad(0))
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, -1.0, 0.0]]
    >>> pa.qwp(np.deg2rad(45))
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, -1.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0]]
    """
    return retarder(np.pi / 2, theta)


def hwp(theta: npt.ArrayLike) -> np.ndarray:
    """Generate Mueller matrix of Half-Wave Plate (HWP)

    Parameters
    ----------
    theta : array_like, (...,)
        Angle of the fast axis

    Returns
    -------
    mueller : np.ndarray, (..., 4, 4)
        Mueller matrix.

    Examples
    --------
    >>> pa.hwp(np.deg2rad(0))
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 0.0, 0.0, -1.0]]
    >>> pa.hwp(np.deg2rad(45))
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, -1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, -1.0]]
    """
    return retarder(np.pi, theta)


def depolarizer(
    d1: npt.ArrayLike = 0.0,
    d2: npt.ArrayLike | None = None,
    d3: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Generate Mueller matrix of the depolarizer.

    Parameters
    ----------
    d1 : float or array_like (...,), optional
        Depolarization factor in [-1, 1] for s1.
        If `d2` and `d3` are None, d1=d2=d3.
    d2 : float or array_like (...,), optional
        Depolarization factor [-1, 1] for s2.
    d3 : float or array_like (...,), optional
        Depolarization factor [-1, 1] for s3.

    Returns
    -------
    mueller : np.ndarray, (..., 4, 4)
        Mueller matrix.

    Examples
    --------
    >>> pa.depolarizer()  # Ideal depolarizer
    [[1. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    >>> pa.depolarizer(0.5)
    [[1.  0.  0.  0. ]
     [0.  0.5 0.  0. ]
     [0.  0.  0.5 0. ]
     [0.  0.  0.  0.5]]
    >>> pa.depolarizer(0.9, 0.8, 0.7)
    [[1.  0.  0.  0. ]
     [0.  0.9 0.  0. ]
     [0.  0.  0.8 0. ]
     [0.  0.  0.  0.7]]
    """
    # Require either 1 arg or 3 args.
    if (d2 is None) ^ (d3 is None):
        raise ValueError("Provide both d2 and d3, or neither.")

    d1 = np.asarray(d1)
    if d2 is None:
        d2 = d3 = d1
    else:
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)

    # Validate range.
    for name, x in (("d1", d1), ("d2", d2), ("d3", d3)):
        if not (np.abs(x) <= 1.0).all():
            raise ValueError(f"Invalid value: {name}={x}. Expected -1 <= {name} <= 1.")

    return _broadcast_mueller(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, d1, 0.0, 0.0],
            [0.0, 0.0, d2, 0.0],
            [0.0, 0.0, 0.0, d3],
        ]
    )


def diattenuator(d: npt.ArrayLike, t: float = 1.0) -> np.ndarray:
    """Generate Mueller matrix of the diattenuator

    Parameters
    ----------
    d : npt.ArrayLike
        Diattenuation vector, (3,)
    t : float, optional
        Transmittance for an unpolarized light, by default 1.0

    Returns
    -------
    np.ndarray
        Mueller matrix, (4, 4)

    Examples
    --------
    Linear polarizer

    >>> 0.5 * pa.diattenuator([1, 0, 0])  # Horizontal linear polarizer (theta = 0)
    [[0.5 0.5 0.  0. ]
     [0.5 0.5 0.  0. ]
     [0.  0.  0.  0. ]
     [0.  0.  0.  0. ]]

    >>> theta = np.deg2rad(30)  # Linear polarizer at 30 degrees
    >>> M1 = 0.5 * pa.diattenuator([np.cos(2 * theta), np.sin(2 * theta), 0])
    >>> M2 = pa.polarizer(theta))
    >>> np.allclose(M1, M2)
    True

    Circular polarizer

    >>> 0.5 * pa.diattenuator([0, 0, 1])  # Right-handed circular polarizer
    [[0.5 0.  0.  0.5]
     [0.  0.  0.  0. ]
     [0.  0.  0.  0. ]
     [0.5 0.  0.  0.5]]

    >>> 0.5 * pa.diattenuator([0, 0, -1])  # Left-handed circular polarizer
    [[ 0.5  0.   0.  -0.5]
     [ 0.   0.   0.   0. ]
     [ 0.   0.   0.   0. ]
     [-0.5  0.   0.   0.5]]
    """
    d = np.asarray(d)
    norm = np.linalg.norm(d, axis=-1)
    d_normalized = d / norm
    m_D = np.sqrt(1 - norm**2) * np.eye(3) + (1 - np.sqrt(1 - norm**2)) * np.outer(d_normalized, d_normalized)
    M_D = np.empty((4, 4))
    M_D[0, 0] = 1
    M_D[0, 1:] = d
    M_D[1:, 0] = d
    M_D[1:, 1:] = m_D
    return t * M_D


ISMUELLER_STOKES = "ISMUELLER_STOKES"  # Stokes criterion by brute-force
ISMUELLER_GK = "ISMUELLER_GK"  # Givens-Kostinski, 1993


def _ismueller_stokes(mueller: npt.ArrayLike, total_size: int = 10000, chunk_size: int = 100) -> np.ndarray:
    """Check physical realizability of Mueller matrix using Stokes criterion by brute-force.

    This function checks the Stokes criterion by projecting
    a dense set of Stokes vectors and verifying that the
    resulting output vectors remain physically valid Stokes vectors.

    To improve efficiency, this function divide the input Stokes vectors
    into small chunks and terminate early, avoiding unnecessary computation.

    Parameters
    ----------
    mueller : array_like
        Mueller matrix of shape (..., 4, 4).
    total_size : int, optional
        Total number of Stokes vectors to test, by default 10000.
    chunk_size : int, optional
        Number of Stokes vectors to test in each chunk, by default 100.

    Returns
    -------
    np.ndarray
        Boolean array of shape (...) indicating whether the Mueller matrix is valid.
    """
    mueller = np.asarray(mueller)  # (..., 4, 4)
    *size, _, _ = mueller.shape
    dtype = mueller.dtype

    is_valid = np.full(size, True, dtype=bool)
    count = 0
    while True:
        # Project random Stokes vectors to the Mueller matrix
        s_in = random.stokes(dop=1.0, size=chunk_size).astype(dtype)  # (chunk_size, 4)
        s_out = np.einsum("...ij,...kj->...ki", mueller[is_valid], s_in, optimize="optimal")  # (..., chunk_size, 4)

        # The output should be valid Stokes vectors
        isstokes = stokes.isstokes(s_out)  # (..., chunk_size)
        is_valid[is_valid] = np.all(isstokes, axis=-1)  # (...)

        # Check the exit condition
        count += chunk_size
        if count >= total_size:
            break
        chunk_size = min(chunk_size, total_size - count)

    return is_valid


def _ismueller_gk(mueller: npt.ArrayLike) -> np.ndarray:
    # Apply eigenvalue decomposition to (G @ M.T @ G @ M)
    M = np.asarray(mueller)  # (..., 4, 4)
    M_T = np.moveaxis(M, -1, -2)  # (..., 4, 4)
    G = np.diag([1.0, -1.0, -1.0, -1.0])  # (4, 4)
    eigenvalues, eigenvectors = np.linalg.eig(G @ M_T @ G @ M)  # (..., 4), (..., 4, 4)

    # All eigenvalues should be real
    is_real = np.all(np.isclose(np.imag(eigenvalues), 0), axis=-1)  # (...,)

    # The eigenvector s_{\sigma_1} corresponding to the largest eigenvalue should itself be a valid Stokes vector
    index = np.argmax(np.abs(eigenvalues), axis=-1)  # (...,)
    stokes_sigma1 = np.take_along_axis(eigenvectors, index[..., None, None], axis=-1).squeeze(-1)  # (..., 4)
    stokes_sigma1 = stokes_sigma1 / stokes_sigma1[..., 0:1]  # Normalize by s0
    is_stokes = stokes.isstokes(stokes_sigma1)  # (...,)

    return is_real & is_stokes  # (...,)



def ismueller(mueller: npt.ArrayLike, method: str = ISMUELLER_GK, **kwargs) -> npt.NDArray[np.bool]:
    """Check physical realizability of Mueller matrix.

    Parameters
    ----------
    mueller : array_like, (..., 4, 4)
        Mueller matrix.
    method : str, optional
        Method to use for checking physical realizability, by default ``pa.ISMUELLER_GK``.

        - ``pa.ISMUELLER_GK``: Givens-Kostinski 1993 [1]_.
        - ``pa.ISMUELLER_STOKES``: Stokes criterion by brute-force.

    Returns
    -------
    is_valid : np.ndarray, (...,)
        Boolean array indicating whether the Mueller matrix is valid.

    References
    ----------
    .. [1] Givens, Clark R., and Alexander B. Kostinski. "A simple necessary and sufficient condition on physically realizable Mueller matrices." Journal of Modern Optics 40.3 (1993): 471-481.
    """
    if method == ISMUELLER_GK:  # Givens-Kostinski, 1993
        return _ismueller_gk(mueller)
    elif method == ISMUELLER_STOKES:  # Stokes criterion by brute-force
        return _ismueller_stokes(mueller, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ISMUELLER_STOKES' or 'ISMUELLER_GK'.")
