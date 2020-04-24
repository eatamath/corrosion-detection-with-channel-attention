import numpy as np
import scipy as sp
from sklearn.metrics import *


def computeMetrics(Ypred:list,Ytest:list):
    Ypred = np.array(Ypred)
    Ytest = np.array(Ytest)
    
    acc = sum(Ypred==Ytest)/len(Ypred)
    
    res = {
        'acc':acc,
    }
    
    return res