import xarray as xr
import numpy as np
from luti import DefaultChecker, Vectorfunction, Vectorset, DefaultInterpolator
from luti.Helpers import xr_reshape

# Idea: Input array in format 2) and invert, repeating the procedure on all additional dimensions.
# Therefore, we cannot avoid apply_ufunc, which is why we have to write a numpy-inversion function
def invert_data_array(data, input_params, output_dim, output_grid, interpolator=DefaultInterpolator(), checker=DefaultChecker()):
    if not all([type(k)==str for k in output_grid.keys()]):
        raise TypeError("The new dimension labels defined in 'output_grid' must be strings.")
    sampledim="d284632842" #create temporary, random dimensions
    newsampledim="d75011395620"

    inv_dims = data[output_dim].values
    inv_dims_labels=[output_grid[g] for g in inv_dims]
    inv_dims_labels_2d=np.array([a.flatten() for a in np.meshgrid(*inv_dims_labels, indexing='ij')]).T
    datam = data.stack({sampledim:input_params})
    input_params_2d = xr.concat([datam[i] for i in input_params], dim="col")
    input_params_2d = input_params_2d.drop_vars(sampledim)
    input_params_2d = input_params_2d.values.T

    def invert_np(sim_params, sim_values, new_values):
        #Inputformat: 2D with [samples, dimensions]
        vf = Vectorfunction(sim_values, sim_params)
        vs = Vectorset(new_values)
        interpolator.initialize(vf)
        checker.initialize(vf)
        new_params=interpolator(vs)
        if checker is not None:
            new_params=checker.check_fill(new_params)
        result=new_params.get_values()
        return result

    data_inv = xr.apply_ufunc(
        lambda new_values: invert_np(input_params_2d, new_values, inv_dims_labels_2d),datam,input_core_dims=[[sampledim, output_dim]],output_core_dims=[[newsampledim, "input_params"]],vectorize=True)
    data_inv = xr_reshape(data_inv,newsampledim, inv_dims, inv_dims_labels)
    data_inv.coords["input_params"]=input_params
    return data_inv
