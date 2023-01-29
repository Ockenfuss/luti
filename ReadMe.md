# Luti - LookUp Tables & Inversion
Facilitating the handling of lookup-tables, including interpolation and inversion.


<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Introduction](#-introduction)
- [Usage](#-usage)
  - [Vectorset and Vectorfunction](#-vectorset-and-vectorfunction)
  - [Interpolating](#-interpolating)
  - [Checking](#-checking)
  - [Scaling](#-scaling)
- [Inversion](#-inversion)
  - [Xarray](#-xarray)

<!-- /code_chunk_output -->


## Introduction
When working with simulation results, based on multiple input parameters, it is often convenient to precompute the simulation for a set of input parameters and store the values in a lookup table for further use.
This module is based on the idea to represent a specific combination of `n` input parameters as a vector in `R^n`. Multiple vectors can be stored in a `Vectorset` object. Similarly, the outcome of a simulation can be seen as a vector `R^m`. The simulation is therefore a function `f: R^n->R^m`, and a lookup table is a discrete representation of this function: `F: R^n:R^m` This representation can be stored in a `Vectorfunction` object. `Luti` provides several methods to interpolate between the discrete points in `Vectorfunction` objects, therefore creating a fast approximation of `f`. 

`Luti` focuses on integration with xarray and provides functionality to create Vectorfunctions from xarray data. Compared to xarray`s built-in interpolation capabilities, Vectorfunctions do not require a regular grid and are therefore more suited when it comes to inversion problems.
## Usage
### Vectorset and Vectorfunction
The most convenient way to create a `Vectorset` is to provide the data in xarray `DataArray` objects. To do so, you must specify one dimension (with length `n`) which contains the coordinates of the vector. All other dimensions in the `DataArray` are flattened out. For a `Vectorfunction`, you must provide an additional `DataArray` of values with a corresponding value dimension (length `m`).
```python
from luti import Vectorset, Vectorfunction
vs=Vectorset.from_dataarray(points, coord_dim='wavelength')
vf=Vectorfunction.from_dataarrays(points, values, coord_dim='wavelength', value_dim='type')
#Alternatively, direct creation from numpy arrays with shape points=[samples, n] and values=[samples,m] is possible:
vs=Vectorset(points)
vf=Vectorfunction(points, values)
```

Internally, vectors and values are stored as 2D numpy arrays. You can access them like
```python
vf.get_points() #shape [samples, n]
vf.get_values() #shape [samples, m]
```

It is possible to convert a `Vectorset` or `Vectorfunction` back into a `DataArray`, but you have to provide an existing `DataArray` as a template.
```python
vs.to_dataarray(similar_to=points, coord_dim='wavelength')
vf.to_dataset(similar_to=points, coord_dim='wavelength', value_dim='type')
```

### Interpolating
Given a `Vectorfunction` with simulation results, you can interpolate it to a `Vectorset` of new input parameters using different provided `Interpolator` objects.
```python
from luti import LinearInterpolator
interpolator=LinearInterpolator(vf) #Uses scipy.interpolate.LinearNDInterpolator for interpolation
vf_interpolated=interpolator(vs)
#Alternatives:
interpolator=NeighbourInterpolator(vf) #Uses nearest neighbours based on sklearn
interpolator=GriddataInterpolator(vf) #Uses scipy.interpolate.griddata interpolation
interpolator=RbfInterpolator(vf) #Uses scipy.interpolate.Rbf for interpolation
```

### Checking
Usually, you do not want to interpolate at input parameters far away from the originally simulated space (e.g. no extrapolation). `Checker` objects allow you to detect such invalid points in a `Vectorset` based on points from another `Vectorset`
```python
from luti import Alphachecker
checker=Alphachecker(vs, alpha=0.1) #Draws a concave hull around the point cloud 'vs' to define the valid area.
mask=checker(vs_new)#Create a mask for a new Vectorset of points
checker.check_fill(vf_interpolated, fill=NaN) #Set invalid points in a vectorfunction to NaN
#Alternatives:
checker=DistanceChecker(vf, threshold=0.1) #points are valid if the (euclidean) distance to a point in 'vf' is at most 'threshold'
checker=BoxChecker(vf) #Check if new points are within the bounding box of the input points, defined by the minimal/maximal extent of the input point cloud in each direction
checker=ConvexHullChecker(vs)#Calculate the convex hull around the point cloud
```
### Scaling
If you have parameters with very different scales, it can be necessary to normalize them before interpolation, depending on the chosen interpolator. Some interpolators like `NeighbourInterpolator` therefore have a `norm` argument.
```python
from luti import NeighbourInterpolator
ni=NeighbourInterpolator(vf, norm=True) #scale all dimensions to [0,1] before any interpolation
from luti import UnitScaler
ni=NeighbourInterpolator(vf, norm=UnitScaler()) #you can explicitly provide a scaler
```
`scikit-learn` provides many different scalers in its preprocessing namespace. You can convert them to `luti Scaler` objects  using the provided `sklearn_scaler_factory`.
```python
from luti import sklearn_scaler_factory
import sklearn as skl
standardScaler=sklearn_scaler_factory(skl.preprocessing.StandardScaler())
ni=NeighbourInterpolator(vf, norm=standardScaler)
```


## Inversion
In science, an often encountered problem is the inversion of a model: Given some measurements and a model, we want to find the input parameters to model the measurements. While a general approach could be e.g. a gradient descent, for a "smooth" model a lookup table can be much faster, especially if the amount of measurements exceeds the set of possible input parameters. In `luti`, such an inversion is straightforward.
```python
import luti
lookup_table=luti.Vectorfunction(simulation_results, simulation_parameters) #This is where the inversion happens
vs_meas=luti.Vectorset(measurements)
#Interpolation
interpolator=luti.NeighbourInterpolator(lookup_table)
input_parameters=interpolator(vs_meas)
#Check
checker=luti.ConvexHullChecker(lookup_table)
checker.check_fill(input_parameters)#set values outside lookup table to NaN
```
> :warning: `luti` checks only if the discrete representation `F:R^n->R^m` is injective, i.e. the inverse is well-defined! However, this does not mean that `f:R^n->R^m` is invertible. You have to check yourself that your original problem does allow for an inversion approach.
### Xarray
The above approach still requires an interpolation for every measurement. For really big datasets, it might be convenient to create a completely inverted lookup table as xarray DataArray, based on a regular grid. At this point, it is instructive to think about the different ways to represent a vectorfunction in xarray

1) **Two DataArrays**, in 2D tabular form:
  Two DataArrays *Parameters[samples, ndim]* and *Values[samples, mdim]*, representing the input parameters and simulated values. This is similar to what is internally done in Vectorfunction objects.

2) **One DataArray**, encoding parameters in the coordinates and values in an extra dim:
  *Data[n1,n2,n3,..., mdim]*. To get the parameters in format *[samples, ndim]*, you need to form the cartesian product between n1xn2xn3... or use 'stack(n1, n2, n3, ...)'. This representation is maybe the most natural form, which you often obtain when evaluating an equation or model over a space of input parameters.

3) **A Dataset with two DataArrays**, representing Parameters and Values of 1).

4) **A Dataset with m Arrays**, representing the Values with names according to the mdim-Dimension in 2):
   `dataset:{ Variables: [m1, m2, m3, ...], Coordinates: [n1, n2, n3, ...]}`

`luti.xarray.invert_data_array` is a high level function to do such an inversion of a DataArray in format 2). Therefore, you have to provide the names of the coordinates belonging to the Parameters (input_params) and the coordinate which describes the simulation output values (output_dim). Additionally, for each label along output_dim, you have to provide a list of values, which will form the regular grid of the inverted table. The following gives a minimal example with `n=m=1`:
```python
import luti
from luti.xarray import invert_data_array
import numpy as np
import xarray as xr

def model(params):
  return 2*params

sim_parameters=np.linspace(0,1,5)
sim_values=model(sim_parameters)
sim_data=xr.DataArray(sim_values,coords=[("sim_params", sim_parameters)])
sim_data=sim_data.expand_dims({"out":["o1"]}) #We need an extra dimension R^m for the simulation output, even if m=1
data_inv=invert_data_array(sim_data, input_params=["sim_params"], output_dim="out", output_grid={"o1":np.linspace(0,2,10)})
```
If there are additional dimensions present, `invert_data_array` repeats the inversion along every dimension not listed in `input_params` or `output_dim`.