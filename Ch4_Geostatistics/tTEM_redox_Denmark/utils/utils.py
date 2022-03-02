import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def point_to_grid(data_point, xmin, ymin, zmin, sx, sy, sz, nx, ny, nz):
    data_grid = np.zeros((nz,ny,nx))
    data_grid[:] = np.nan
    data_point_rescale = np.array(np.floor((data_point[:,:3]-[xmin,ymin,zmin])/[sx,sy,sz]),dtype = 'int64')
    data_grid[data_point_rescale[:,2],data_point_rescale[:,1],data_point_rescale[:,0]] = data_point[:,-1]
    return data_grid


def save_gslib(data_point_DF,save_dir = None, name = 'logrho'):
    if save_dir is None:  
        save_dir = name+'.gslib'
    else: 
        save_dir = save_dir+name+'.gslib'
    
    data_point_DF_name = data_point_DF.columns
    data_point = data_point_DF.values
    data_point[np.isnan(data_point)] = -999.
    num_var = np.int(data_point.shape[1])
    
    with open(save_dir, "ab") as f:
        temp = name+'\n'
        f.write(temp.encode())
        temp = str(num_var)+'\n'
        f.write(temp.encode())
        
        for i in range(num_var):
            temp = data_point_DF_name[i]+'\n'
            f.write(temp.encode())
        np.savetxt(f, data_point, fmt='%1.2f')