from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from tqdm import tqdm

def surrounding_tTEM_multiple_borehole(tTEM,redox_loc_list,bz = 1,by = 1, bx = 1):
    nearby_tTEM = np.zeros((redox_loc_list.shape[0],(bz*2+1)*(by*2+1)*(bx*2+1)))
    tTEM_pad = np.pad(tTEM, ((bz, bz), (by, by), (bx, bx)), 'edge')
    ny_ext = tTEM_pad.shape[1]
    nx_ext = tTEM_pad.shape[2]
    
    # vectorize
    tTEM_pad = tTEM_pad.reshape(-1)
    t = 0
    for i in tqdm(np.arange(-bz,bz+1)):
        for j in np.arange(-by,by+1):
            for k in np.arange(-bx,bx+1):
                nearby_tTEM[:,t] = tTEM_pad[(redox_loc_list[:,0]+bz+i)*ny_ext*nx_ext +(redox_loc_list[:,1]+by+j)*nx_ext+redox_loc_list[:,2]+bx+k]
                t = t+1
    
    return nearby_tTEM


def LR_redox(tTEM_sgsim,grid_mask,redox_grid):
    # tTEM info, near the borehole (60m 60m 4m)
    bz = 4;by = 3; bx = 3
    redox_loc = np.array(np.where(tTEM_sgsim)).T
    nearby_tTEM = surrounding_tTEM_multiple_borehole(tTEM_sgsim*grid_mask,
                                                     redox_loc,bz = bz,by = by,bx = bx)
    
    # Fill in the NA by mean
    row_mean = np.nanmean(nearby_tTEM,axis = 1)
    inds = np.where(np.isnan(nearby_tTEM))
    nearby_tTEM[inds] = np.take(row_mean, inds[0])
    
    # PCA of the surrounding
    sample_idx = np.random.choice(np.where(~np.isnan(nearby_tTEM[:,0]))[0],1000)
    pca = PCA(n_components=nearby_tTEM.shape[1])
    pca.fit(nearby_tTEM[sample_idx,:])
    
    
    # Obtain X, y
    redox_borehole = np.array(np.where(~np.isnan(redox_grid))).T
    nearby_tTEM_borehole = surrounding_tTEM_multiple_borehole(tTEM_sgsim*grid_mask,redox_borehole,bz = bz,by = by,bx = bx)
    y = np.array(redox_grid[~np.isnan(redox_grid)],dtype = 'int64')

    row_mean = np.nanmean(nearby_tTEM_borehole,axis = 1)
    inds = np.where(np.isnan(nearby_tTEM_borehole))
    nearby_tTEM_borehole[inds] = np.take(row_mean, inds[0])
    
    # Only training on the borehole with relative certain surrounding data
    training_idx = np.unique(np.where(~np.isnan(nearby_tTEM_borehole))[0])
    
    X = np.hstack([pca.transform(nearby_tTEM_borehole[training_idx,:])[:,:40],
               np.array(DEM_int[np.where(~np.isnan(redox_grid))[1],
                        np.where(~np.isnan(redox_grid))[2]]-np.where(~np.isnan(redox_grid))[0])[training_idx].reshape(-1,1)])
    
    clf = LogisticRegression(random_state=10, solver='lbfgs').fit(X, y[training_idx])
    y_pred = clf.predict(X)
    
    # Accuracy
    acc = accuracy_score(y[training_idx],y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y[training_idx],y_pred)
    
    # Inference
    # Apply/Predict on both area
    DEM_multiple = np.zeros(sgsim_mean.shape)
    DEM_multiple = DEM_multiple+DEM_int
    test_idx_all = np.unique(np.where(~np.isnan(nearby_tTEM[:,0:1]))[0])
    
    y_test = np.zeros(nx*ny*nz)
    y_test[:] = np.nan

    start = 0
    sep = 50000
    for end in np.arange(0,test_idx_all.shape[0],sep)+sep:
        test_idx = test_idx_all[start:end]
        test_pc_scores = pca.transform(nearby_tTEM[test_idx,:])[:,:40]
        X_test = np.hstack([test_pc_scores,
                            np.array(DEM_multiple.reshape(-1)-np.where(sgsim_mean)[0].reshape(-1))[test_idx].reshape(-1,1)])
        y_test[test_idx] = clf.predict(X_test)
        start = end

    y_test = np.array(y_test.reshape(nz,ny,nx),dtype = 'float64')
    
    return y_test*grid_mask, pca, clf, acc, conf_matrix, X