import numpy as np
import scipy as sp
from sklearn.metrics import *
import sklearn.metrics as metrics


def computeMetrics(Ypred:list,Ytest:list):
    
    Ypred = np.array(Ypred)
    Ytest = np.array(Ytest)

    acc = metrics.accuracy_score(Ypred,Ytest)
    f_score_weighted = metrics.f1_score(Ytest, Ypred, average='weighted')
    f_score_macro = metrics.f1_score(Ytest, Ypred, average='macro')
    jaccrd_score = metrics.jaccard_similarity_score(Ytest,Ypred)
    ham_distance = metrics.hamming_loss(Ytest,Ypred)
    kappa = metrics.cohen_kappa_score(Ytest,Ypred,labels=[1,2,3,4])

    scores = {}
    scores['acc'] = acc
    scores['f_score'] = [f_score_weighted, f_score_macro]
    scores['jaccrd_score'] = jaccrd_score
    scores['ham_distance'] = ham_distance
    scores['kappa'] = kappa
    
    return scores