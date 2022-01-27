
## author: David zhen Yin, yinzhen@stanford.edu
## modified by Lijing Wang, lijing52@stanford.edu (2021.2)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
def loadings_plot(fa_loadings, loading_names, var_exp_prop, n_fctr = 4):
  # fa_loadings: 2D array [n_factor, n_loadings]
  # n_fctr: number factors to plot
  # var_exp_prop: proportions of variance explained

  ## get loadings numbers and loading name
  n_loading = fa_loadings.shape[1]

  ## divide x-axis by variance explained proportions
  # var_prop = variance_explained/variance_explained.sum()
  x_axis = 100*np.sqrt(var_exp_prop[:n_fctr])

  prop_cusum = x_axis.cumsum()

  ## Making arrays with [x-ticks, loadings] for all loading
  note_names = [] # including loading names for plot annotation
  xticks = [] # make ticks of x-axis 
  xticks2 = ['0%'] # make ticks of secondary x-axis  as variances proportions
  xaxis_loadings = np.zeros((n_fctr*n_loading, 2))

  for i in range(n_fctr):
    xaxis_loadings[int(n_loading*i):int(n_loading*(i+1)), 0] = np.linspace(x_axis[i]*0.1, x_axis[i]*0.9, n_loading) \
                                                                + np.cumsum(np.r_[0,x_axis])[i]  

    # loading_scaled = fa_loadings[i]/np.linalg.norm(fa_loadings[i])
    # xaxis_loadings[n_loading*(i):n_loading*(i+1), 1] = loading_scaled
    xaxis_loadings[n_loading*(i):n_loading*(i+1), 1] = fa_loadings[i]
    
    # add loading names
    note_names = note_names + loading_names
    xticks.append('F'+str(i+1))
    # add variance explained
    xticks2.append(str(round(var_exp_prop.cumsum()[i]*100)) +'%')

  ## make loading plot
  fig = plt.figure(figsize=(20,8))
  
  ax = fig.add_subplot(111)
  for idx in range(len(xaxis_loadings[:,1])):
      if np.abs(xaxis_loadings[idx,1])>0.3:
          ax.scatter(xaxis_loadings[idx,0], xaxis_loadings[idx,1], s=50, c='C1')
      else:
          ax.scatter(xaxis_loadings[idx,0], xaxis_loadings[idx,1], s=10, c='C0')
  ax2 = ax.twiny()
  for i in range(len(xaxis_loadings)):
      
      plt.annotate(note_names[i], 
                  (xaxis_loadings[i,0], xaxis_loadings[i,1]-0.05), 
                  fontsize = 11)
  # make vertical lines to separate factors
  plt.vlines(x=np.cumsum(x_axis[:-1]), ymin=-1, ymax=1), plt.vlines(x=np.sum(x_axis)+2, ymin=-1, ymax=1)
  plt.hlines(y=0, xmin=-5, xmax=x_axis.sum()+1)
  plt.hlines(y=[0.3, -0.3], xmin=-5, xmax=x_axis.sum()+1, linestyles='--', linewidth=0.5)
  
  # y-axis labels and ticks
  plt.ylim(-1, 1), ax.set_ylabel('Loadings', fontsize=14), ax.yaxis.set_tick_params(labelsize=13)
  
  # x-axis labels and ticks
  ax.set_xlim(0, np.sum(x_axis)+1), ax2.set_xlim(0, np.sum(x_axis)+1)
  ax2.set_xticks(x_axis/2+np.cumsum(np.r_[0,x_axis])[:-1]), ax2.set_xticklabels(xticks, fontsize=13)

  ax.set_xticks(np.cumsum(np.r_[0,x_axis])), ax.set_xticklabels(xticks2, fontsize=12)
  ax.set_xlabel('variance explained', fontsize=14)
  
  plt.show()

  return 
