from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from statsmodels.discrete.discrete_model import MNLogit
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def LR_redox(tTEM_sgsim,grid_mask,redox_grid,DEM_int,nx,ny,nz):
    # tTEM info, near the borehole (60m 60m 4m)
    bz = 4;by = 3; bx = 3
    redox_loc = np.array(np.where(tTEM_sgsim)).T
    nearby_tTEM = surrounding_tTEM_multiple_borehole(tTEM_sgsim*grid_mask,
                                                     redox_loc,bz = bz,by = by,bx = bx)
    
    nearby_tTEM[np.isnan(nearby_tTEM)] = 0
    
    # PCA of the surrounding
    sample_idx = np.random.choice(np.where(~np.isnan(nearby_tTEM[:,0]))[0],1000)
    pca = PCA(n_components=nearby_tTEM.shape[1])
    pca.fit(nearby_tTEM[sample_idx,:])
    
    
    # Obtain X, y
    redox_borehole = np.array(np.where(~np.isnan(redox_grid))).T
    nearby_tTEM_borehole = surrounding_tTEM_multiple_borehole(tTEM_sgsim*grid_mask,redox_borehole,bz = bz,by = by,bx = bx)
    y = np.array(redox_grid[~np.isnan(redox_grid)],dtype = 'int64')

    nearby_tTEM_borehole[np.isnan(nearby_tTEM_borehole)] = np.nanmean(tTEM_sgsim)

    
    # logistic regression
    X = np.hstack([pca.fit_transform(nearby_tTEM_borehole)[:,:40],
               np.array(DEM_int[np.where(~np.isnan(redox_grid))[1],
                        np.where(~np.isnan(redox_grid))[2]]-np.where(~np.isnan(redox_grid))[0]).reshape(-1,1)])

    # fit the logistic regression model 
    logit_model = MNLogit(y,X)
    logit_fit = logit_model.fit()

    y_pred_prob = logit_fit.predict(X)
    y_pred = np.argmax(y_pred_prob,axis = 1)

    # Accuracy
    acc = accuracy_score(y,y_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y,y_pred)

    # Inference
    # Apply/Predict on both area
    DEM_multiple = np.zeros((nz,ny,nx))
    DEM_multiple = DEM_multiple+DEM_int
    DEM_multiple = DEM_multiple.reshape(-1)

    y_test = np.zeros((nx*ny*nz,3))
    y_test[:] = np.nan

    start = 0
    sep = 50000
    for end in tqdm(np.arange(0,nx*ny*nz,sep)+sep):
        if end>=nx*ny*nz:
            end = nx*ny*nz
        test_idx = np.arange(start,end)
        test_pc_scores = pca.transform(nearby_tTEM[test_idx,:])[:,:40]
        X_test = np.hstack([test_pc_scores,
                            DEM_multiple[test_idx].reshape(-1,1)])
        y_test[test_idx,:] = logit_fit.predict(X_test)
        start = end

    y_test = np.array(y_test.reshape(nz,ny,nx,3),dtype = 'float64')

    return y_test*grid_mask.reshape(nz,ny,nx,1), pca, logit_fit, acc, conf_matrix, X, y, y_pred_prob


def precision_recall(y_pred_prob,threshold,y):
    prediction = (y_pred_prob[:,0]>threshold)
    TP = np.sum(prediction[y==0]) # 
    PP = np.sum(prediction) # predicted positive
    P = np.sum(y==0)# positive
    precision = TP/PP
    if np.isnan(precision):
        precision = 1
    recall = TP/P
    return [precision,recall]


def precision_recall_plot(y_pred_prob,y):
    label_list = ['0: reduced', '1: reducing', '2: oxic']
    for cat in range(3):
        PR = np.array([precision_recall(y_pred_prob[:,cat:(cat+1)],threshold, (y!=cat)*1) for threshold in np.linspace(0,1,100)])
        plt.plot(PR[:,1],PR[:,0],'-', label = label_list[cat] )
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')


def confusion_matrix_plot(conf_matrix):
    label_list = ['0: reduced', '1: reducing', '2: oxic']
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=label_list)
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax= ax, cmap = 'Blues')