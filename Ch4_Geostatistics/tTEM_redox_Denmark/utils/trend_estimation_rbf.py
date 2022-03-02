import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from tqdm import tqdm

# 3D bounding box of this area (~5km^2)
## x: UTM (m)
## y: UTM (m)
## z: masl. (m)
xmin = 548100
ymin = 6220100
zmin = 20
xmax = 550480
ymax = 6222200
zmax = 109

# sx, xy, xz: resolution of the grid (play with this in Exercise 1)
sx = 20
sy = 20
sz = 1

# nx, ny, nz: num of each dimension in this grid
nx = np.int((xmax-xmin)/sx)
ny = np.int((ymax-ymin)/sy)
nz = np.int((zmax-zmin)/sz)

def trend_estimation_rbf(tTEM_grid, num_point_trend = 500):
    z,y,x = np.where(~np.isnan(tTEM_grid))
    logrho = tTEM_grid[np.where(~np.isnan(tTEM_grid))]

    sample_idx = np.random.choice(len(logrho),num_point_trend,replace = False)
    rbfi = Rbf(z[sample_idx], y[sample_idx], x[sample_idx], logrho[sample_idx])  # radial basis function interpolator instance

    xi = np.arange(nx)
    yi = np.arange(ny)
    zi = np.arange(nz)
    zi,yi,xi = np.meshgrid(zi,yi,xi,indexing='ij')
    logrhoi = rbfi(zi, yi, xi)   # interpolated values
    
    return logrhoi

def multiple_trend_rbf(tTEM_grid, num_trend = 100):
    multiple_trend = np.zeros((nx*ny*nz,num_trend))
    for i in tqdm(range(num_trend)):
        logrhoi = trend_estimation_rbf(tTEM_grid)
        multiple_trend[:,i] = logrhoi.reshape(-1)
    return multiple_trend


