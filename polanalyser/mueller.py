import numpy as np

def calcMueller(intensities, muellers_p, muellers_a):
    """Calculate Mueller matrix from observed intensities and Mueller matrixes of light source and analyzer
    
    Parameters
    ----------
    intensities : np.ndarray
        Measured intensities. (height, width, K)
    muellers_p : np.ndarray
        Mueller matrix of the light source. (3, 3, K) or (4, 4, K)
    muellers_a : np.ndarray
        Mueller matrix of the analyzer. (3, 3, K) or (4, 4, K)

    Returns
    -------
    mueller : np.ndarray
        Mueller matrix. (height, width, 9) or (height, width, 16)
    """
    if not isinstance(intensities, np.ndarray):
        intensities = np.stack(intensities, axis=-1)  # (height, width, K)

    if not isinstance(muellers_p, np.ndarray):
        muellers_p = np.stack(muellers_p, axis=-1)  # (3, 3, K) or (4, 4, K)

    if not isinstance(muellers_a, np.ndarray):
        muellers_a = np.stack(muellers_a, axis=-1)  # (3, 3, K) or (4, 4, K)
    
    K = intensities.shape[-1]
    D = muellers_p.shape[0]  # scalar: 3 or 4
    W = np.empty((K, D*D))
    for k in range(K):
        P1 = np.expand_dims(muellers_p[:, 0, k], axis=1)  # [m11, m21, m31] or [m11, m21, m31, m41]
        A1 = np.expand_dims(muellers_a[0, :, k], axis=0)  # [m11, m12, m13] or [m11, m12, m13, m14]
        W[k] = np.ravel((P1 @ A1).T)

    W_pinv = np.linalg.pinv(W)  # (D*D, K)
    mueller = np.tensordot(W_pinv, intensities, axes=(1, -1)) # (9, height, width) or (16, height, width)
    mueller = np.moveaxis(mueller, 0, -1) # (height, width, 9) or ((height, width, 16)
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
    theta : float
      the angle of rotation

    Returns
    -------
    mueller : np.ndarray
      mueller matrix (4, 4)
    """
    return rotator(-theta) @ mueller @ rotator(theta)

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


def plotMueller(filename, img_mueller, vabsmax=None, dpi=300, cmap="RdBu", add_title=True):
    """
    Apply color map to the Mueller matrix image and save them side by side
    
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
        Whether to insert a title (e.g. m11, m12...) in the image.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    try:
        plt.rcParams["mpl_toolkits.legacy_colorbar"] = False
    except KeyError:
        pass
    
    # Check for 'img_muller' shape
    height, width, channel = img_mueller.shape
    if channel==9:
        n = 3
    elif channel==16:
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
            maintitle = "$m$$_{0}$$_{1}$".format(i//n+1, i%n+1) # m{}{}
            t = add_inner_title(ax, maintitle, loc='lower right')
        
        # Add image
        im = ax.imshow(img_mueller[:,:,i],
                       vmin=vmin,
                       vmax=vmax,
                       cmap=cmap,
                       )

    # Colorbar
    cbar = ax.cax.colorbar(im, ticks=[vmin, 0, vmax])
    cbar.solids.set_edgecolor("face")
    ax.cax.toggle_label(True)
    
    # Save figure
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
