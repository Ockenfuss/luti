import xarray as xr
import numpy as np
from luti import DefaultChecker, Vectorfunction, Vectorset, DefaultInterpolator
from luti.Helpers import xr_reshape

# Idea: Input array in format 2) and invert, repeating the procedure on all additional dimensions.
# Therefore, we cannot avoid apply_ufunc, which is why we have to write a numpy-inversion function
def invert_data_array(data, input_dims, output_dim, output_grid, interpolator=DefaultInterpolator(), checker=DefaultChecker()):
    newdims = data[output_dim].values
    ynew = np.array([output_grid[o] for o in newdims]).T
    datam = data.stack(samples=input_dims)
    xold = xr.concat([datam[i] for i in input_dims], dim="col")
    xold = xold.drop_vars("samples")
    xold = xold.values.T

    def invert_np(xold, yold, ynew):
        # xold: [samples, ndim], y: [samples,mdim]
        vf = Vectorfunction(yold, xold)
        vs = Vectorset(ynew)
        interpolator.initialize(vf)
        checker.initialize(vf)
        xnew=interpolator(vs)
        if checker is not None:
            xnew=checker.check_fill(xnew)
        result=xnew.get_values()
        return result

    data_inv = xr.apply_ufunc(
        lambda yold: invert_np(xold, yold, ynew),datam,input_core_dims=[["samples", output_dim]],output_core_dims=[["newsamples", "input_dim"]],vectorize=True)
    print(newdims)
    print(ynew)
    data_inv = xr_reshape(data_inv, "newsamples", newdims, ynew.T)
    return data_inv
