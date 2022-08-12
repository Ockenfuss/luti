from numpy.testing._private.utils import IgnoreException
import xarray as xr
import numpy as np
import alphashape
import shapely.vectorized
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import Rbf, griddata, LinearNDInterpolator, interp1d
from scipy.spatial import Delaunay
from luti.Helpers import ShapeError, DimensionError


class Vectorset(object):
    def __init__(self, points):
        self.points = np.atleast_2d(points.T).T
        self.samples=self.points.shape[0]
        self.ndim=self.points.shape[1]

    @classmethod
    def from_dataarray(cls, data, coord_dim=None):
        if coord_dim is None:
            return cls(data.stack(new=[...]).values)
        if coord_dim not in data.dims:
            raise ValueError(f"Dimension {coord_dim} not found in DataArray!")
        return cls(data.stack(new=[d for d in data.dims if d != coord_dim]).transpose(..., coord_dim).values)

    def get_points(self):
        return self.points

    def __str__(self):
        return "Points:\n" + repr(self.points)

    @staticmethod
    def _np_to_dataarray(nparr, similar_to, coord_dim=None):

        original = similar_to.dims
        if coord_dim is not None:
            if coord_dim not in similar_to.dims:
                similar_to = similar_to.expand_dims({coord_dim: np.arange(nparr.shape[1])})
                original = similar_to.dims
            similar_to = similar_to.transpose(..., coord_dim)
        return similar_to.copy(data=nparr.reshape(similar_to.shape)).transpose(*original)

    def to_dataarray(self, similar_to, coord_dim=None):
        return self._np_to_dataarray(self.get_points(), similar_to, coord_dim)


class Vectorfunction(Vectorset):
    def __init__(self, points, values=None):
        super().__init__(points)
        if values is not None:
            self.values = np.atleast_2d(values.T).T
            assert(len(self.values) == len(self.points))
            self.mdim=self.values.shape[1]

    @classmethod
    def from_dataarrays(cls, data_coord: xr.DataArray, data_val: xr.DataArray, coord_dim: str = None, value_dim: str = None):
        dims1 = [d for d in data_coord.dims if d != coord_dim]
        dims2 = [d for d in data_val.dims if d != value_dim]
        try:
            assert(np.all([d in dims2 for d in dims1]))
            assert(np.all([d in dims1 for d in dims2]))
        except AssertionError:
            raise DimensionError(f"Coordinate and value array must have the same flattened dimensions, but these dimensions are {dims1} and {dims2}!")
        try:
            assert(np.all([data_val[d].size == data_coord[d].size for d in data_val.dims if d != value_dim]))
            assert(np.all([data_val[d].size == data_coord[d].size for d in data_coord.dims if d != coord_dim]))
        except AssertionError:
            raise ShapeError("data_coord and data_val must have the same shape (except in coord_dim and value_dim)!")
        return cls(super().from_dataarray(data_coord, coord_dim).get_points(), super().from_dataarray(data_val, value_dim).get_points())

    @classmethod
    def from_vectorset(cls, vectorset: Vectorset, values: np.ndarray):
        return cls(vectorset.get_points(), values)

    def get_values(self):
        return self.values

    def to_dataset(self, similar_to: xr.DataArray, coord_dim: str = None, value_dim: str = None):
        similar_to_points = similar_to
        if value_dim in similar_to.dims:
            similar_to_points = similar_to.isel(drop=True, **{value_dim: 0})
        da_points = self.to_dataarray(similar_to_points, coord_dim)
        da_points.name = "Points"
        similar_to_values = similar_to
        if coord_dim in similar_to.dims:
            similar_to_values = similar_to.isel(drop=True, **{coord_dim: 0})
        da_values = self._np_to_dataarray(self.get_values(), similar_to_values, value_dim)
        da_values.name = "Values"
        return xr.Dataset({"Points": da_points, "Values": da_values})

    def __str__(self):
        return super().__str__() + '\n\nValues:\n' + repr(self.values)


def normfactors(vectorset: Vectorset):
    return np.amax(vectorset.get_points(), axis=0) - np.amin(vectorset.get_points(), axis=0)


class Checker(object):
    def __init__(self, vectorset: Vectorset = None):
        raise(NotImplementedError)

    def initialize(self, vectorset: Vectorset):
        self.ndim=vectorset.ndim

    def check(self, vectorset: Vectorset) -> np.ndarray:
        if vectorset.ndim!=self.ndim:
            raise ValueError(f"Can not check {vectorset.ndim}-D data because initialization happened with {self.ndim}-D data.")

    def check_fill(self, vectorfunction: Vectorfunction, fill=np.nan):
        mask = self.check(vectorfunction)
        values = vectorfunction.get_values()
        values[~mask, :] = fill
        return Vectorfunction(vectorfunction.get_points(), values)

    def __call__(self, vectorset: Vectorset):
        return self.check(vectorset)


class Alphachecker(Checker):
    def __init__(self, vectorset: Vectorset = None, alpha: float = 0.0):
        self.alpha=alpha
        if vectorset is not None:
          self.initialize(vectorset)

    def initialize(self, vectorset: Vectorset):
        super().initialize(vectorset)
        if vectorset.get_points().ndim != 2 or self.ndim != 2:
            raise ValueError("Alphashapes are available for more than 2 dimensions since vers. 1.3.0., but the 'check' function needs to be implemented.")
        self.alphashape = alphashape.alphashape(vectorset.get_points(), self.alpha)

    def check(self, vectorset: Vectorset):
        super().check(vectorset)
        points = vectorset.get_points()
        return shapely.vectorized.contains(self.alphashape, points[:, 0], points[:, 1])


class Distancechecker(Checker):
    def __init__(self, vectorset: Vectorset = None, threshold: float=1.0, norm=True):
        self.threshold = threshold
        self.norm=norm
        if vectorset is not None:
          self.initialize(vectorset)
    
    def initialize(self, vectorset: Vectorset):
        super().initialize(vectorset)
        if self.norm:
            self.normfactor = normfactors(vectorset)
        else:
            self.normfactor = 1
        self.nbs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(vectorset.get_points() / self.normfactor)

    def check(self, vectorset: Vectorset):
        super().check(vectorset)
        distance, indices = self.nbs.kneighbors(vectorset.get_points() / self.normfactor)
        return distance[:, 0] < self.threshold

class BoxChecker(Checker):
    """Check if checkpoints are within the bounding box, defined by the maximum/minimum extend of the pointcloud in every direction.
    """
    def __init__(self, vectorset: Vectorset = None):
        if vectorset is not None:
            self.initialize(vectorset)

    def initialize(self, vectorset: Vectorset):
        super().initialize(vectorset)
        self.mins=vectorset.get_points().min(axis=0)
        self.maxs=vectorset.get_points().max(axis=0)
    
    def check(self, vectorset: Vectorset):
        super().check(vectorset)
        mask=(vectorset.get_points()>=self.mins ) * (vectorset.get_points()<=self.maxs)
        return np.all(mask, axis=1)

class ConvexHullChecker(Checker):
    """Calculate the convex hull around the pointcloud, using scipy.Delaunay.
    For 1D data, use a BoxChecker to check only min/max values.
    """
    def __init__(self, vectorset: Vectorset = None):
        if vectorset is not None:
            self.initialize(vectorset)
        
    def initialize(self, vectorset: Vectorset):
        super().initialize(vectorset)
        if self.ndim==1:
            self.boxchecker=BoxChecker(vectorset)
        else:
            self.hull=Delaunay(vectorset.get_points())
    
    def check(self, vectorset: Vectorset):
        super().check(vectorset)
        if self.ndim==1:
            return self.boxchecker(vectorset)
        return self.hull.find_simplex(vectorset.get_points())>=0

class DefaultChecker(ConvexHullChecker):
    def __init__(self, vectorset: Vectorset = None):
        super().__init__(vectorset)
  



class Interpolator(object):
    def __init__(self, vectorfunction: Vectorfunction = None):
        raise NotImplementedError
    
    def initialize(self, vectorfunction: Vectorfunction):
        raise NotImplementedError

    def interp(self, vectorset: Vectorset) -> Vectorfunction:
        raise NotImplementedError

    def __call__(self, vectorset: Vectorset):
        return self.interp(vectorset)


class RbfInterpolator(Interpolator):
    def __init__(self, vectorfunction: Vectorfunction = None, norm=True):
      self.norm=norm
      if vectorfunction is not None:
        self.initialize(vectorfunction)

    def initialize(self, vectorfunction: Vectorfunction):
        self.normfactor = 1
        if self.norm:
            self.normfactor = normfactors(vectorfunction)
        self.interpolators = [Rbf(*(vectorfunction.get_points() / self.normfactor).T, d, function='linear', smooth=0.0) for d in vectorfunction.get_values().T]

    def interp(self, vectorset: Vectorset):
        values = np.array([interp(*(vectorset.get_points() / self.normfactor).T) for interp in self.interpolators])
        return Vectorfunction(vectorset.get_points(), values.T)


class NeighbourInterpolator(Interpolator):
    def __init__(self, vectorfunction: Vectorfunction = None, norm=True):
      self.norm=norm
      if vectorfunction is not None:
        self.initialize(vectorfunction)

    def initialize(self, vectorfunction: Vectorfunction):
        self.normfactor=1
        if self.norm:
            self.normfactor = normfactors(vectorfunction)
        self.nbs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(vectorfunction.get_points() / self.normfactor)
        self.values = vectorfunction.get_values()
        self.points = vectorfunction.get_points()

    def interp(self, vectorset: Vectorset):
        indices = self.nbs.kneighbors(vectorset.get_points() / self.normfactor, return_distance=False)[:, 0]
        return Vectorfunction(self.points[indices, :], self.values[indices, :])


class GriddataInterpolator(Interpolator):
    def __init__(self, vectorfunction: Vectorfunction = None, norm=True):
        """Alternative to InterpolationMatcher based on Scipy.girddata interpolation. This is for compatibility with older retrievals, InterpolationMatcher (based on scipy.LinearNDInterpolator) is generally to be preferred, since it allows to store the interpolator between multiple calls.

        Args:
            vectorfunction ([Vectorfunction]): Simulation data as Vectorfunction.
            norm (bool, optional): Normalize all dimensions to cover a range between 0 and 1. Defaults to True.
        """
        self.norm=norm
        if vectorfunction is not None:
          self.initialize(vectorfunction)

    def initialize(self, vectorfunction: Vectorfunction):
        self.normfactor = 1
        if self.norm:
            self.normfactor = normfactors(vectorfunction)
        self.vf = vectorfunction

    def interp(self, vectorset: Vectorset):
        values = np.array([griddata(self.vf.get_points() / self.normfactor, val, vectorset.get_points() / self.normfactor) for val in self.vf.get_values().T])
        return Vectorfunction.from_vectorset(vectorset, values.T)

class LinearInterpolator(Interpolator):
    def __init__(self, vectorfunction: Vectorfunction = None, norm=True):
        self.norm=norm
        if vectorfunction is not None:
          self.initialize(vectorfunction)

    def initialize(self, vectorfunction: Vectorfunction):
        self.normfactor = 1
        self.ndim=vectorfunction.ndim
        if self.norm:
            self.normfactor = normfactors(vectorfunction)
        norm_points=vectorfunction.get_points() / self.normfactor
        if self.ndim==1:
            self.interpolator = interp1d(norm_points[:,0], vectorfunction.get_values(), axis=0) #NdInterpolator works only for >1 dimensions
        else:
            self.interpolator = LinearNDInterpolator(norm_points, vectorfunction.get_values())

    def interp(self, vectorset: Vectorset):
        norm_points=vectorset.get_points() / self.normfactor
        assert(vectorset.ndim==self.ndim)
        if self.ndim==1:
            norm_points=norm_points[:,0]
        values= self.interpolator(norm_points)
        return Vectorfunction(vectorset.get_points(), values)

#Define a default matcher used as a default in other luti functions
class DefaultInterpolator(LinearInterpolator):
    def __init__(self, vectorfunction: Vectorfunction = None, norm=True):
        super().__init__(vectorfunction, norm)

