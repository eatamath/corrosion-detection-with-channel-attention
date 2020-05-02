import numpy as np
from scipy.ndimage import *
import matplotlib.pyplot as plt

import seaborn as sns

def plotResultCurve(_metrics:list,att_names:list,title=''):
    res = np.array(list(map(lambda x:[x[a] for a in att_names],_metrics)))
    ax = plt.figure()
    
    handles = []
    for i,a in enumerate(att_names,0):
        line, = plt.plot(res[:,i],label=a)
        handles.append(line)
        
    plt.title(title)
    legend = plt.legend(handles=handles,loc=4)
    plt.show()
    
    return