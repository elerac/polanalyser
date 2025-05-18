from pathlib import Path
from typing import Union, Tuple
import numpy as np
import numpy.typing as npt
from .io import load


def directions_to_rusinkiewicz(wi: np.ndarray, wo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert directions to Rusinkiewicz coordinates.

    This function is originally from Mitsuba 3, and modified to work with numpy arrays.
    https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/bsdfs/measured_polarized.cpps
    """
    wi = wi / np.linalg.norm(wi, axis=-1, keepdims=True)
    wo = wo / np.linalg.norm(wo, axis=-1, keepdims=True)

    h = wi + wo
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)

    n = np.array([0.0, 0.0, 1.0])

    b = np.cross(n, h)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)

    t = np.cross(b, h)
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)

    theta_d = np.arccos(np.sum(h * wi, axis=-1))
    theta_h = np.arccos(np.sum(n * h, axis=-1))

    i_prj = wi - np.sum(wi * h, axis=-1, keepdims=True) * h
    i_prj = i_prj / np.linalg.norm(i_prj, axis=-1, keepdims=True)

    cos_phi_d = np.sum(t * i_prj, axis=-1)
    sin_phi_d = np.sum(b * i_prj, axis=-1)

    phi_d = np.arctan2(sin_phi_d, cos_phi_d)

    return phi_d, theta_d, theta_h


class MeasuredPolarimetricBRDF:
    """Utility class for accessing a measured pBRDF table.

    To use this class, download the pBRDF dataset [Baek et al., SIGGRAPH2020] from the following link:
    https://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html

    Examples
    --------
    Access the pBRDF table:

    >>> measured_pbrdf = pa.MeasuredPolarimetricBRDF("2_white_billiard_mitsuba/2_white_billiard_inpainted.pbsdf")
    >>> wi = np.array([0.342, 0, 0.94])  # (3,)
    >>> wo = np.array([-0.337, 0.059, 0.94])  # (3,)
    >>> wvl = 550
    >>> M = measured_pbrdf(wi, wo, wvl)  # (4, 4)
    >>> M
    [[ 0.742  0.156 -0.004 -0.005]
     [ 0.24   0.683 -0.052 -0.001]
     [-0.188 -0.057 -0.626 -0.011]
     [-0.001 -0.009  0.002 -0.548]]

    Access the pBRDF table for multiple wavelengths:

    >>> wvl = [450, 550, 650] # (3,)
    >>> M = measured_pbrdf(wi, wo, wvl)  # (3, 4, 4)

    Access the pBRDF table for multiple directions and wavelengths as a image:

    >>> height, width = 240, 320
    >>> wi = np.full((height, width, 1, 3), wi)  # (height, width, 1, 3)
    >>> wo = np.full((height, width, 1, 3), wo)  # (height, width, 1, 3)
    >>> wvl = [450, 550, 650]  # (3,)
    >>> M = measured_pbrdf(wi, wo, wvl)  # (height, width, 3, 4, 4)
    """

    def __init__(self, filepath_pbsdf: Union[str, Path]):
        pbrdf_table = load(filepath_pbsdf)
        self.M = pbrdf_table["M"]  # (361, 91, 91, 5, 4, 4)
        self.phi_d = np.atleast_1d(pbrdf_table["phi_d"].squeeze())  # (361,)
        self.theta_d = np.atleast_1d(pbrdf_table["theta_d"].squeeze())  # (91,)
        self.theta_h = np.atleast_1d(pbrdf_table["theta_h"].squeeze())  # (91,)
        self.wvls = np.atleast_1d(pbrdf_table["wvls"].squeeze())  # (5,)

        # Check is sorted
        is_sorted_phi_d = np.all(np.diff(self.phi_d) >= 0).item()
        is_sorted_theta_d = np.all(np.diff(self.theta_d) >= 0).item()
        is_sorted_theta_h = np.all(np.diff(self.theta_h) >= 0).item()
        is_sorted_wvls = np.all(np.diff(self.wvls) >= 0).item()
        is_sorted = is_sorted_phi_d and is_sorted_theta_d and is_sorted_theta_h and is_sorted_wvls
        if not is_sorted:
            raise ValueError("The input data (phi_d, theta_d, theta_h, wvls) must be sorted in ascending order.")

    def __call__(self, wi: npt.ArrayLike, wo: npt.ArrayLike, wvl: npt.ArrayLike) -> np.ndarray:
        """Look up the Mueller matrix for given incident and outgoing directions and wavelength.

        Parameters
        ----------
        wi : array_like, (..., 3)
            Incident direction vector.
        wo : array_like, (..., 3)
            Outgoing direction vector.
        wvl : float or array_like, (...,)
            Wavelength.

        Returns
        -------
        M_sampled : np.ndarray, (..., 4, 4)
            Mueller matrix for the given incident and outgoing directions and wavelength.
        """
        wi = np.asarray(wi)
        wo = np.asarray(wo)
        wvl = np.asarray(wvl)

        if wi.shape != wo.shape:
            raise ValueError("wi and wo must have the same shape")

        phi_d, theta_d, theta_h = directions_to_rusinkiewicz(wi, wo)

        phi_d_idx = self._argmin(self.phi_d, phi_d)
        theta_d_idx = self._argmin(self.theta_d, theta_d)
        theta_h_idx = self._argmin(self.theta_h, theta_h)
        wvl_idx = self._argmin(self.wvls, wvl)

        M_sampled = self.M[phi_d_idx, theta_d_idx, theta_h_idx, wvl_idx]
        return M_sampled

    @staticmethod
    def _argmin(a: np.ndarray, a_query: np.ndarray) -> np.ndarray:
        """Find the index of the closest value in a sorted 1D array."""
        idx_1 = np.searchsorted(a, a_query, side="right") - 1
        idx_2 = np.clip(idx_1 + 1, 0, len(a) - 1)
        idx = np.where(np.abs(a[idx_1] - a_query) < np.abs(a[idx_2] - a_query), idx_1, idx_2)
        return idx


def main():
    np.set_printoptions(suppress=True, precision=3)

    filepath_pbsdf = "pbsdf/2_white_billiard_mitsuba/2_white_billiard_inpainted.pbsdf"
    pbrdf = MeasuredPolarimetricBRDF(filepath_pbsdf)

    theta_i = np.deg2rad(20)
    phi_i = np.deg2rad(0)
    theta_o = np.deg2rad(20)
    phi_o = np.deg2rad(170)
    sph2cart = lambda theta, phi: np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    wi = sph2cart(theta_i, phi_i)
    wo = sph2cart(theta_o, phi_o)
    wvl = 550

    print(f"wi: {wi}")
    print(f"wo: {wo}")
    print(f"wvl: {wvl}")

    M = pbrdf(wi, wo, wvl)
    print(M.shape)
    print(M[0])


if __name__ == "__main__":
    main()
