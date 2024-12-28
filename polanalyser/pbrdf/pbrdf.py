from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from polanalyser.pbrdf.io import load_pbsdf


class PolarimetricBRDF(ABC):
    """Abstract class for utilising pBRDF."""

    @abstractmethod
    def __call__(self, wi: np.ndarray, wo: np.ndarray, wvl: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate BRDF.

        Parameters
        ----------
        wi : np.ndarray, (..., 3)
            Incident direction.
        wo : np.ndarray, (..., 3)
            Outgoing direction.
        wvl : np.ndarray, (...,)
            Wavelength.

        Returns
        -------
        m : np.ndarray, (..., 4, 4)
            pBRDF in Mueller matrix form.
        """
        raise NotImplementedError


def directions_to_rusinkiewicz(wi: np.ndarray, wo: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert directions to Rusinkiewicz coordinates.

    This function is originally from Mitsuba 3, and modified to work with numpy arrays.
    https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/bsdfs/measured_polarized.cpps
    """
    wi = wi / np.linalg.norm(wi, axis=-1, keepdims=True)
    wo = wo / np.linalg.norm(wo, axis=-1, keepdims=True)

    h = wi + wo
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)

    n = np.array([0.0, 0.0, 1.0])[None, :]

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


class MeasuredPolarimetricBRDF(PolarimetricBRDF):
    """Utility class for accessing pBRDF data

    This class provides functionality for working with tabulated pBRDF data as described in [Baek et al., SIGGRAPH2020].

    To use this class, download the pBRDF dataset from the following link:
    https://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html

    Examples
    --------
    >>> pbrdf = MeasuredPolarimetricBRDF("2_white_billiard_mitsuba/2_white_billiard_inpainted.pbsdf")
    >>> wi = np.array([0.342, 0, 0.94])
    >>> wo = np.array([-0.337, 0.059, 0.94])
    >>> wvl = 550
    >>> M = pbrdf(wi, wo, wvl)  # (1, 4, 4)
    >>> M
    [[ 0.742  0.156 -0.004 -0.005]
     [ 0.24   0.683 -0.052 -0.001]
     [-0.188 -0.057 -0.626 -0.011]
     [-0.001 -0.009  0.002 -0.548]]
    """

    def __init__(self, filepath_pbsdf: Union[str, Path]):
        pbrdf = load_pbsdf(filepath_pbsdf)
        self.M = pbrdf["M"]  # (361, 91, 91, 5, 4, 4)
        self.phi_d = pbrdf["phi_d"].squeeze()  # (361,)
        self.theta_d = pbrdf["theta_d"].squeeze()  # (91,)
        self.theta_h = pbrdf["theta_h"].squeeze()  # (91,)
        self.wvls = pbrdf["wvls"].squeeze()  # (5,)

    def __call__(self, wi: np.ndarray, wo: np.ndarray, wvl: Union[float, np.ndarray]) -> np.ndarray:
        wvl = np.array(wvl)

        # Convert directions to Rusinkiewicz coordinates
        phi_d, theta_d, theta_h = directions_to_rusinkiewicz(wi, wo)

        # Find closest index
        phi_d_idx = np.argmin(np.abs(np.angle(np.exp(1j * (self.phi_d - phi_d[..., None])))), axis=-1)
        theta_d_idx = np.argmin(np.abs(self.theta_d - theta_d[..., None]), axis=-1)
        theta_h_idx = np.argmin(np.abs(self.theta_h - theta_h[..., None]), axis=-1)
        wvl_idx = np.argmin(np.abs(self.wvls - wvl[..., None]), axis=-1)

        # Sample pBRDF
        M_sampled = self.M[phi_d_idx, theta_d_idx, theta_h_idx, wvl_idx]

        return M_sampled


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
