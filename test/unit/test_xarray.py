import unittest as ut
import numpy.testing as npt
import xarray as xr
import numpy as np
import luti as lut
import luti.xarray as lux


class TestXarray(ut.TestCase):
    def test_invert_data_array_2d(self):
        """Test using a R^2->R^2 linear, invertible function."""
        def f(d):
        #matrix vector multiplication in xarray: sum([out, in] * [in], d=in)
            matrix=xr.DataArray([[2,1], [6,4]], dims=["out","in"], coords={"out":("out", ["o1", "o2"])})
            return (matrix*d).sum(dim="in")
        def f_inv(d):
            matrix_inv=xr.DataArray([[2,-0.5], [-3,1]], dims=["input_params", "in"])
            return (matrix_inv*d).sum(dim="in")

        nsamples=50
        p1=np.linspace(-50,60,nsamples)
        p2=np.linspace(-90,100,nsamples)
        p1m, p2m=np.meshgrid(p1, p2,indexing='ij')
        parameters=xr.DataArray([p1m, p2m], dims=["in", "p1", "p2"], coords={"in":("in",["i1","i2"]),"p1":p1, "p2":p2})
        values=f(parameters)
        grid={"o1":np.linspace(0,30,31), "o2":np.linspace(0,100,101)}
        params_inv=lux.invert_data_array(values, input_params=["p1", "p2"], output_dim="out", output_grid=grid, checker=lut.ConvexHullChecker(), interpolator=lut.LinearInterpolator())
        params_inv=params_inv.transpose("input_params", "o1", "o2").round(2)

        #check: use inverse matrix to calculate values explicitly
        o1m, o2m=np.meshgrid(grid["o1"], grid["o2"],indexing='ij')
        values_check=xr.DataArray([o1m, o2m], dims=["in", "o1", "o2"])
        params_check=f_inv(values_check)
        params_check=params_check.where(~np.isnan(params_inv))
        data_diff=params_inv-params_check
        data_diff.plot(x="o1", y="o2", col="input_params")
        xr.testing.assert_allclose(params_inv, params_check)