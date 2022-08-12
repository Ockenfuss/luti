import pandas as pd


class DimensionError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class ShapeError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


def remove_nondim(da):
    nondim = [d for d in list(da.coords.keys()) if d not in da.dims]
    return da.drop_vars(nondim)


# This function  is currently missing in xarray (https://github.com/pydata/xarray/issues/2419)
def xr_reshape(A, dim, newdims, coords):
    """Reshape DataArray A to convert its dimension dim into sub-dimensions given by
    newdims and the corresponding coords.
    Example: Ar = xr_reshape(A, 'time', ['year', 'month'], [(2017, 2018), np.arange(12)])"""

    # Create a pandas MultiIndex from these labels
    ind = pd.MultiIndex.from_product(coords, names=newdims)

    # Replace the time index in the DataArray by this new index,
    A1 = A.copy()

    A1.coords[dim] = ind

    # Convert multiindex to individual dims using DataArray.unstack().
    # This changes dimension order! The new dimensions are at the end.
    A1 = A1.unstack(dim)

    # Permute to restore dimensions
    i = A.dims.index(dim)
    dims = list(A1.dims)

    for d in newdims[::-1]:
        dims.insert(i, d)
    for d in newdims:
        _ = dims.pop(-1)

    return A1.transpose(*dims)
