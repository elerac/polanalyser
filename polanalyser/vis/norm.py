import numpy as np
from matplotlib import colors


class SymPowerNorm(colors.CenteredNorm):
    def __init__(self, gamma, halfrange=None, clip=False):
        """
        Normalize symmetric data with a sign aware power-law.

        This class is like a combination of `PowerNorm` and `CenteredNorm`.
        It is beneficial for enhancing contrast in data that is symmetric around zero.

        Parameters
        ----------
        gamma : float
            Power law exponent.
        halfrange : float, optional
            The range of data values that defines a range of ``0.5`` in the
            normalization, so that *vcenter* - *halfrange* is ``0.0`` and
            *vcenter* + *halfrange* is ``1.0`` in the normalization.
            Defaults to the largest absolute difference to *vcenter* for
            the values in the dataset.
        clip : bool, default: False
            Determines the behavior for mapping values outside the range
            ``[vmin, vmax]``.

            If clipping is off, values outside the range ``[vmin, vmax]`` are
            also transformed, resulting in values outside ``[0, 1]``.  This
            behavior is usually desirable, as colormaps can mark these *under*
            and *over* values with specific colors.

            If clipping is on, values below *vmin* are mapped to 0 and values
            above *vmax* are mapped to 1. Such values become indistinguishable
            from regular boundary values, which may cause misinterpretation of
            the data.

        Examples
        --------
        >>> norm = SymPowerNorm(gamma=1 / 2.2, halfrange=1.0)
        >>> plt.imshow(img, cmap="RdBu", norm=norm)
        """
        super().__init__(vcenter=0.0, halfrange=halfrange, clip=clip)
        self.gamma = float(gamma)

    def __call__(self, value, clip=None):
        result = super().__call__(value, clip=clip)
        data = result.data
        data = 2.0 * data - 1.0
        data = np.sign(data) * np.power(np.abs(data), self.gamma)
        data = 0.5 + 0.5 * data
        result = np.ma.array(data, mask=result.mask, copy=False)
        return result

    def inverse(self, value):
        value = 2.0 * value - 1.0
        value = np.sign(value) * np.power(np.abs(value), 1.0 / self.gamma)
        value = 0.5 * value + 0.5
        value = super().inverse(value)
        return value
