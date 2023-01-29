import unittest as ut
import numpy as np
import numpy.testing as npt
import xarray as xr
import xarray.testing as xrt
import luti as lut


class TestVectorset(ut.TestCase):
    def setUp(self):
        self.cubic = xr.DataArray(np.arange(8).reshape(2, 2, 2), coords=[("x", ["a", "b"]), ("y", [0, 1]), ("z", ["c", "d"])])

    def test_vectorset_from_dataarray(self):
        vs = lut.Vectorset.from_dataarray(self.cubic, 'x')
        npt.assert_equal(vs.get_points()[:, 0], [0, 1, 2, 3])
        vs = lut.Vectorset.from_dataarray(self.cubic, 'y')
        npt.assert_equal(vs.get_points()[:, 0], [0, 1, 4, 5])
        npt.assert_equal(vs.get_points()[:, 1], [2, 3, 6, 7])
        vs = lut.Vectorset.from_dataarray(self.cubic, 'z')
        npt.assert_equal(vs.get_points()[:, 0], [0, 2, 4, 6])

        npt.assert_raises(ValueError, lut.Vectorset.from_dataarray, self.cubic, 'w')

    def test_vectorset_from_dataarray_1d(self):
        da = xr.DataArray(np.arange(4).reshape(2, 2), coords=[("x", ["a", "b"]), ("y", [0, 1])]).transpose('y', 'x')
        vs = lut.Vectorset.from_dataarray(da)
        npt.assert_equal(vs.get_points()[:, 0], [0, 1, 2, 3])

    def test_dataarray_from_vectorset(self):
        vs = lut.Vectorset.from_dataarray(self.cubic, 'y')
        new = vs.to_dataarray(self.cubic, coord_dim='y')
        xrt.assert_equal(self.cubic, new)
        vs = lut.Vectorset.from_dataarray(self.cubic, 'x')
        new = vs.to_dataarray(self.cubic, coord_dim='x')
        xrt.assert_equal(self.cubic, new)
        vs = lut.Vectorset.from_dataarray(self.cubic, 'z')
        new = vs.to_dataarray(self.cubic, coord_dim='z')
        xrt.assert_equal(self.cubic, new)

        # Insert coord_dim if not existent
        vs = lut.Vectorset.from_dataarray(self.cubic, 'y')
        new = vs.to_dataarray(self.cubic.isel(y=0, drop=True), coord_dim='w')
        assert('w' in new.dims)
        npt.assert_equal(new['w'].values, [0, 1])
        xrt.assert_equal(self.cubic, new.rename({'w': 'y'}).transpose('x', 'y', 'z'))

    def test_dataarray_from_vectorset_1d(self):
        vs=lut.Vectorset.from_dataarray(self.cubic)
        new = vs.to_dataarray(self.cubic)
        xrt.assert_equal(self.cubic, new)


class TestVectorfunction(ut.TestCase):
    def setUp(self):
        self.points_cubic = xr.DataArray(np.arange(8).reshape(2, 2, 2), coords=[("x", ["a", "b"]), ("y", [0, 1]), ("z", ["c", "d"])])
        self.values_cubic = xr.DataArray(np.arange(8).reshape(2, 2, 2), coords=[("x", ["a", "b"]), ("y", [0, 1]), ("z", ["c", "d"])])

    def test_initialiaze_vectorfunction(self):
        values_square = self.values_cubic.isel(y=0, drop=True)
        vf = lut.Vectorfunction.from_dataarrays(self.points_cubic, values_square, 'y')
        npt.assert_equal(vf.get_points()[:, 0], [0, 1, 4, 5])
        npt.assert_equal(vf.get_points()[:, 1], [2, 3, 6, 7])
        npt.assert_equal(vf.get_values()[:, 0], [0, 1, 4, 5])

        vf = lut.Vectorfunction.from_dataarrays(self.points_cubic, self.values_cubic, 'y', 'y')
        npt.assert_equal(vf.get_points()[:, 0], [0, 1, 4, 5])
        npt.assert_equal(vf.get_points()[:, 1], [2, 3, 6, 7])
        npt.assert_equal(vf.get_values()[:, 0], [0, 1, 4, 5])
        npt.assert_equal(vf.get_values()[:, 1], [2, 3, 6, 7])

        npt.assert_raises(lut.Helpers.DimensionError, lut.Vectorfunction.from_dataarrays, self.points_cubic, self.values_cubic, 'y', 'z')
        npt.assert_raises(lut.Helpers.DimensionError, lut.Vectorfunction.from_dataarrays, self.points_cubic, self.values_cubic, 'y')
        npt.assert_raises(lut.Helpers.DimensionError, lut.Vectorfunction.from_dataarrays, self.points_cubic, values_square.isel(x=0), 'y')
        npt.assert_raises(lut.Helpers.DimensionError, lut.Vectorfunction.from_dataarrays, self.points_cubic, values_square.isel(x=0), 'y')

        values_more = xr.DataArray(np.arange(6).reshape(2, 3), coords=[("x", ["a", "b"]), ("y", [0, 1, 2])])
        npt.assert_raises(lut.Helpers.ShapeError, lut.Vectorfunction.from_dataarrays, self.points_cubic, values_more, 'z')

    def test_dataarray_from_vectorfunction(self):
        values_cubic = self.values_cubic.rename({'y': 'w'})
        vf = lut.Vectorfunction.from_dataarrays(self.points_cubic, values_cubic, 'y', 'w')
        xrt.assert_equal(vf.to_dataset(values_cubic, 'y', 'w').Points.transpose('x', 'y', 'z'), self.points_cubic)
        xrt.assert_equal(vf.to_dataset(values_cubic, 'y', 'w').Values, values_cubic)
        xrt.assert_equal(vf.to_dataset(self.points_cubic, 'y', 'w').Points, self.points_cubic)
        xrt.assert_equal(vf.to_dataset(self.points_cubic, 'y', 'w').Values.transpose('x', 'w', 'z'), values_cubic)

    def test_dataarray_from_vectorfunction_1d(self):
        vf=lut.Vectorfunction.from_dataarrays(self.points_cubic, self.values_cubic)
        new=vf.to_dataset(self.points_cubic)
        xrt.assert_equal(self.points_cubic, new.Points)
        xrt.assert_equal(self.values_cubic, new.Values)

    def test_copy(self):
        vf=lut.Vectorfunction(np.array([1,2]), np.array([1,2]))
        vfc=vf #no copy
        vf.values[0,0]=2
        assert(vfc.values[0,0]==2)
        vfc=vf.copy() #deep copy
        vf.values[0,0]=3
        assert(vfc.values[0,0]==2)
    
    def test_is_injective(self):
        vf=lut.Vectorfunction(np.ones((3,3)), np.arange(9).reshape(3,3))
        assert(vf.is_injective()==True)
        vf=lut.Vectorfunction(np.ones((3,3)), np.ones((3,3)))
        assert(vf.is_injective()==False)

    def test_is_welldefined(self):
        vf=lut.Vectorfunction(np.arange(9).reshape(3,3), np.ones((3,3)))
        assert(vf.is_welldefined()==True)
        vf=lut.Vectorfunction(np.ones((3,3)), np.ones((3,3)))
        assert(vf.is_welldefined()==False)

class TestScaler(ut.TestCase):
    def test_unitscaler(self):
        vs=lut.Vectorset(np.array([-3,-1,1,2]))
        scaler=lut.UnitScaler()
        npt.assert_allclose(scaler.fit_transform(vs).get_points()[:,0], [0,2/5, 4/5,1]) #values between 0 and 1
        vs=lut.Vectorset(np.array([-5,3])) #now test the initialized scaler
        npt.assert_allclose(scaler.transform(vs).get_points()[:,0], [-2/5, 6/5]) #values outside 0 and 1



class TestChecker(ut.TestCase):
    def setUp(self):
        # the example from https://alphashape.readthedocs.io/en/latest/readme.html
        points = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.5, 0.25], [0.5, 0.75], [0.25, 0.5], [0.75, 0.5]])
        self.vs2d = lut.Vectorset(points)
        points=np.array([0,0.3,0.6,0.9,1.0])
        self.vs1d=lut.Vectorset(points)

    def test_boxchecker(self):
        #1D
        checker=lut.BoxChecker(self.vs1d)
        testset = lut.Vectorset(np.array([-1,0,0.5,1,1.5]))
        mask=checker(testset)
        npt.assert_equal(mask, [False, True, True, True, False])
        #2D
        points=np.array([[0,1], [1,0], [2,1], [1,2]])#45deg turned square
        vs=lut.Vectorset(points)
        checker=lut.BoxChecker(vs)
        testset=lut.Vectorset(np.array([[0,0], [-0.1,0.5], [0.1,0.1], [1.1, 1.1], [2.1,1.1]]))
        mask=checker(testset)
        npt.assert_equal(mask, [True, False, True, True, False])


    def test_alphachecker(self):
        ack = lut.Alphachecker(self.vs2d, 0.1)
        testpoints = np.array([[0.5, 0.5], [0, 0.5], [0.8, 0.2], [1, 0.5], [0.5, 0], [0.5, 1]])
        testset = lut.Vectorset(testpoints)
        mask = ack.check(testset)
        npt.assert_equal(mask, [True, False, True, False, False, False])

    def test_convexhull_checker(self):
        #2D
        checker = lut.ConvexHullChecker(self.vs2d)
        testpoints = np.array([[0.5, 0.5], [-0.1, 0.5], [0.8, 0.2], [1, 0.5],[0.5, 0.3], [0.5, -1e-10], [0.5, 1], [1.2,0.5]])
        testset = lut.Vectorset(testpoints)
        mask = checker.check(testset)
        npt.assert_equal(mask, [True, False, True, True, True, False, True, False])
        #1D: simple min/max check
        checker=lut.ConvexHullChecker(self.vs1d)
        testset = lut.Vectorset(np.array([-1,0,0.5,1,1.5]))
        mask=checker(testset)
        npt.assert_equal(mask, [False, True, True, True, False])
        


    def test_distancechecker(self):
        # Scale by factor 2 in y
        testvs = lut.Vectorset(self.vs2d.get_points() * np.array([1, 2]))
        testpoints = np.array([[1, 2], [1, 2.1], [1, 2.2], [1.1, 2], [1.05, 2.1], [1.07, 2.14], [1.08, 2.15]])
        testset = lut.Vectorset(testpoints)
        dck = lut.Distancechecker(testvs, threshold=0.1)
        mask = dck.check(testset)
        npt.assert_equal(mask, [True, True, False, False, True, True, False])

        dck = lut.Distancechecker(testvs, threshold=0.1, norm=False)
        mask = dck.check(testset)
        npt.assert_equal(mask, [True, False, False, False, False, False, False])
        dck = lut.Distancechecker(testvs, threshold=0.1, norm=lut.IdentityScaler()) #explicitly provide scaler
        mask = dck.check(testset)
        npt.assert_equal(mask, [True, False, False, False, False, False, False])


class TestInterpolators(ut.TestCase):
    def setUp(self) -> None:
        #2D diagonal plane
        points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        values = np.array([1, 0, 0, -1])
        self.vf = lut.Vectorfunction(points, values)

        values = np.repeat(values[:, np.newaxis], 2, 1)
        self.vf2 = lut.Vectorfunction(points, values)

        #1D diagonal line
        points = np.array([0,0.5,1])
        values = 2*points
        self.vf1d= lut.Vectorfunction(points, values)

    def test_linearinterpolator(self):
        # Issue: (0.5,0) not on 0.5 in interpolation!
        im = lut.LinearInterpolator(self.vf)
        testpoints = np.array([[0, 0], [0.5, 0.5], [0.2, 0.8]])
        testset = lut.Vectorset(testpoints)
        result = im.interp(testset)
        npt.assert_allclose(result.get_values()[:, 0], [1, 0, 0], rtol=0.05, atol=1e-7)

        im = lut.LinearInterpolator(self.vf2)
        result = im.interp(testset)
        npt.assert_allclose(result.get_values()[:, 0], [1, 0, 0], rtol=0.05, atol=1e-7)
        npt.assert_allclose(result.get_values()[:, 1], [1, 0, 0], rtol=0.05, atol=1e-7)

        #1D
        im = lut.LinearInterpolator(self.vf1d)
        testpoints=np.array([0,0.5,0.7])
        vs= lut.Vectorset(testpoints)
        result = im.interp(vs)
        npt.assert_allclose(result.get_values()[:, 0], [0,1,1.4], rtol=0.05, atol=1e-7)

    def test_radialinterpolator(self):
        im = lut.RbfInterpolator(self.vf)
        testpoints = np.array([[0, 0], [0.5, 0.5], [0.2, 0.8]])
        testset = lut.Vectorset(testpoints)
        result = im.interp(testset)
        npt.assert_allclose(result.get_values()[:, 0], [1, 0, 0], rtol=0.05, atol=1e-7)

        im = lut.RbfInterpolator(self.vf2)
        result = im.interp(testset)
        npt.assert_allclose(result.get_values()[:, 0], [1, 0, 0], rtol=0.05, atol=1e-7)
        npt.assert_allclose(result.get_values()[:, 1], [1, 0, 0], rtol=0.05, atol=1e-7)

    def test_neighbourinterpolator(self):
        nm = lut.NeighbourInterpolator(self.vf2)
        testpoints = np.array([[0, 0], [0.5, 0], [0.25, 0.25], [0.75, 0.75], [1.25, 0]])
        testset = lut.Vectorset(testpoints)
        result = nm.interp(testset)
        npt.assert_equal(result.get_points(), testpoints) #the points stay the same
        npt.assert_equal(result.get_values()[:, 0], [1, 1, 1, -1, 0])
        npt.assert_equal(result.get_values()[:, 1], [1, 1, 1, -1, 0])

    def test_neighbourinterpolator_constant(self):
        #Check if interpolation works with a constant value as well
        vf=lut.Vectorfunction(np.arange(3)*0.5, np.ones(3)*2) #well defined constant function
        im=lut.NeighbourInterpolator(vf)
        testpoints = np.array([1,2,3])
        testset = lut.Vectorset(testpoints)
        result = im.interp(testset)
        npt.assert_allclose(result.get_values()[:, 0], [2,2,2])

        vf=lut.Vectorfunction(np.ones(3)*0.5, np.arange(3)*2)# function not well defined (one x value points to multiple y)
        with self.assertRaises(ValueError):
            lut.NeighbourInterpolator(vf)

    def test_gridddatainterpolator(self):
        im = lut.GriddataInterpolator(self.vf)
        testpoints = np.array([[0, 0], [0.5, 0.5], [0.2, 0.8]])
        testset = lut.Vectorset(testpoints)
        result = im.interp(testset)
        npt.assert_allclose(result.get_values()[:, 0], [1, 0, 0], rtol=0.05, atol=1e-7)

        im = lut.GriddataInterpolator(self.vf2)
        result = im.interp(testset)
        npt.assert_allclose(result.get_values()[:, 0], [1, 0, 0], rtol=0.05, atol=1e-7)
        npt.assert_allclose(result.get_values()[:, 1], [1, 0, 0], rtol=0.05, atol=1e-7)

