from typing import List, Optional
import numpy as np

def calcMueller(intensities: List[np.ndarray], muellers_psg: List[np.ndarray], muellers_psa: List[np.ndarray]):
    """Calculate Mueller matrix from observed intensities and Mueller matrixes of Polarization State Generator (PSG) and Polarization State Analyzer (PSA)

    This function calculates Mueller matrix image from images captured under a variety of polarimetric conditions (both PSG and PSA).
    Polarimetric conditions are described by Mueller matrix form (`muellers_psg` and `muellers_psa`).

    The unknown Mueller matrix is calculated by the least-squares method from pairs of intensities and Muller matrices.
    The number of input pairs must be greater than the number of Mueller matrix parameters (i.e., more than 9 or 16).
    
    Parameters
    ----------
    intensities : List[np.ndarray]
        Measured intensities.
    muellers_psg : List[np.ndarray]
        Mueller matrix of the Polarization State Generator (PSG). (3, 3) or (4, 4)
    muellers_psa : List[np.ndarray]
        Mueller matrix of the Polarization State Analyzer (PSA). (3, 3) or (4, 4)

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix. (height, width, 9) or (height, width, 16)
    """
    lists_length = [len(intensities), len(muellers_psg), len(muellers_psa)]
    if not all(x == lists_length[0] for x in lists_length):
        raise ValueError(f"The length of the list must be the same, not {lists_length}.")

    # Convert List[np.ndarray] to np.ndarray
    intensities = np.stack(intensities, axis=-1)  # (height, width, K)
    muellers_psg = np.stack(muellers_psg, axis=-1)  # (3, 3, K) or (4, 4, K)
    muellers_psa = np.stack(muellers_psa, axis=-1)  # (3, 3, K) or (4, 4, K)
    
    K = intensities.shape[-1]  # scalar
    D = muellers_psg.shape[0]  # sclar: 3 or 4
    W = np.empty((K, D*D))
    for k in range(K):
        P1 = np.expand_dims(muellers_psg[:, 0, k], axis=1)  # [m00, m10, m20] or [m00, m10, m20, m30]
        A1 = np.expand_dims(muellers_psa[0, :, k], axis=0)  # [m00, m01, m02] or [m00, m01, m02, m03]
        W[k] = np.ravel((P1 @ A1).T)

    W_pinv = np.linalg.pinv(W)  # (D*D, K)
    mueller = np.tensordot(W_pinv, intensities, axes=(1, -1))  # (9, height, width) or (16, height, width)
    mueller = np.moveaxis(mueller, 0, -1)  # (height, width, 9) or ((height, width, 16)
    return mueller

def rotator(theta):
    """Generate Mueller matrix of rotation

    Parameters
    ----------
    theta : float
      the angle of rotation

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    ones = np.ones_like(theta)
    zeros = np.zeros_like(theta)
    sin2 = np.sin(2*theta)
    cos2 = np.cos(2*theta)
    mueller = np.array([[ones,  zeros, zeros, zeros],
                  [zeros,  cos2,  sin2, zeros],
                  [zeros, -sin2,  cos2, zeros],
                  [zeros, zeros, zeros, ones]])
    mueller = np.moveaxis(mueller, [0,1], [-2,-1])
    return mueller

def rotateMueller(mueller, theta):
    """Rotate Mueller matrix
    
    Parameters
    ----------
    mueller : np.ndarray
      mueller matrix (3, 3) or (4, 4)
    theta : float
      the angle of rotation

    Returns
    -------
    mueller_rotated : np.ndarray
      rotated mueller matrix (3, 3) or (4, 4)
    """
    if mueller.shape == (4, 4):
        mueller_rotated = rotator(-theta) @ mueller @ rotator(theta)
        return mueller_rotated
    elif mueller.shape == (3, 3):
        mueller4x4 = np.zeros((4, 4), dtype=mueller.dtype)
        mueller4x4[:3, :3] = mueller
        mueller_rotated4x4 = rotateMueller(mueller4x4, theta)
        mueller_rotated = mueller_rotated4x4[:3, :3]
        return mueller_rotated

def polarizer(theta):
    """Generate Mueller matrix of linear polarizer

    Parameters
    ----------
    theta : float
      the angle of the linear polarizer

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    mueller = np.array([[0.5, 0.5, 0, 0],
                  [0.5, 0.5, 0, 0],
                  [  0,   0, 0, 0],
                  [  0,   0, 0, 0]]) # (4, 4)
    mueller = rotateMueller(mueller, theta)
    return mueller

def retarder(delta, theta):
    """Generate Mueller matrix of linear retarder
    
    Parameters
    ----------
    delta : float
      the phase difference between the fast and slow axis
    theta : float
      the angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    ones = np.ones_like(delta)
    zeros = np.zeros_like(delta)
    sin = np.sin(delta)
    cos = np.cos(delta)
    mueller = np.array([[ones,  zeros, zeros, zeros],
                        [zeros, ones,  zeros, zeros],
                        [zeros, zeros, cos,   -sin],
                        [zeros, zeros, sin,   cos]])
    mueller = np.moveaxis(mueller, [0,1], [-2,-1])
    
    mueller = rotateMueller(mueller, theta)
    return mueller

def qwp(theta):
    """Generate Mueller matrix of Quarter-Wave Plate (QWP)
    
    Parameters
    ----------
    theta : float
      the angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    return retarder(np.pi/2, theta)

def hwp(theta):
    """Generate Mueller matrix of Half-Wave Plate (QWP)
    
    Parameters
    ----------
    theta : float
      the angle of the fast axis

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    return retarder(np.pi, theta)


def plotMueller(filename: str, img_mueller: np.ndarray, vabsmax: Optional[float] = None, dpi: float = 300, cmap: str = "RdBu", add_title: bool = True):
    """Apply color map to the Mueller matrix image and save them side by side
    
    Parameters
    ----------
    filename : str
        File name to be written.
    img_mueller : np.ndarray, (height, width, 9) or (height, width, 16)
        Mueller matrix image.
    vabsmax : float
        Absolute maximum value for plot. If None, the absolute maximum value of 'img_mueller' will be applied.
    dpi : float
        The resolution in dots per inch.
    cmap : str
        Color map for plot.
        https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    add_title : bool
        Whether to insert a title (e.g. m00, m01...) in the image.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    # Check for 'img_muller' shape
    height, width, channel = img_mueller.shape
    if channel == 9:
        n = 3
    elif channel == 16:
        n = 4
    else:
        raise ValueError(f"'img_mueller' shape should be (height, width, 9) or (height, width, 16): ({height}, {width}, {channel})")
    
    def add_inner_title(ax, title, loc, size=None, **kwargs):
        """
        Insert the title inside image
        """
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patheffects import withStroke

        if size is None:
            size = dict(size=plt.rcParams['legend.fontsize'])

        at = AnchoredText(title, loc=loc, prop=size,
                          pad=0., borderpad=0.5,
                          frameon=False, **kwargs)

        ax.add_artist(at)
        at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
        return at
        
    # Vreta figure
    fig = plt.figure()

    # Create image grid
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n,n),
                     axes_pad=0.0,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="3%",
                     cbar_pad=0.10,
                     )
    
    # Set absolute maximum value
    vabsmax = np.max(np.abs(img_mueller)) if (vabsmax is None) else vabsmax
    vmax =  vabsmax
    vmin = -vabsmax

    # Add data to image grid
    for i, ax in enumerate(grid):
        # Remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        if add_title:
            # maintitle = "$m$$_{0}$$_{1}$".format(i//n+1, i%n+1) # m{}{}
            maintitle = f"$m$$_{i//n}$$_{i%n}$" # m{}{}
            _ = add_inner_title(ax, maintitle, loc='lower right')
        
        # Add image
        im = ax.imshow(img_mueller[:,:,i], vmin=vmin, vmax=vmax, cmap=cmap)

    # Colorbar
    cbar = ax.cax.colorbar(im, ticks=[vmin, 0, vmax])
    cbar.solids.set_edgecolor("face")
    ax.cax.toggle_label(True)
    
    # Save figure
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)