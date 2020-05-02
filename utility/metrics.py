import numpy as np
import scipy as sp
from sklearn.metrics import *


def computeMetrics(Ypred:list,Ytest:list):
    Ypred = np.array(Ypred)
    Ytest = np.array(Ytest)
    
    N = len(Ypred)
    
    scores = {}

    acc = metrics.accuracy_score(Ypred,Ytest)
    
    auc = metrics.roc_auc_score(Ypred,Ytest)
    
    fpr, tpr, thresholds = metrics.roc_curve(Ytest, Ypred, pos_label=1)
    
    mcc = metrics.matthews_corrcoef(Ytest, Ypred)
    
    tn = sum((Ypred==0) & (Ytest==0))
    fp = sum((Ypred==1) & (Ytest==0))
    tnr = float(tn)/(fp+tn)
    
    tp = sum((Ypred==1) & (Ytest==1))
    ppv = float(tp)/(tp+fp)
    
    f_score = metrics.f1_score(Ytest, Ypred)
    
    ap = metrics.average_precision_score(Ytest, Ypred)
    
    bacc = metrics.balanced_accuracy_score(Ytest, Ypred)
    
    brier = metrics.brier_score_loss(Ytest, Ypred, pos_label=1)
    
    recall = metrics.recall_score(Ytest, Ypred)
    
    scores['acc'] = acc
    scores['auc'] = auc
    scores['fpr'] = fpr
    scores['tpr'] = tpr
    scores['mcc'] = mcc
    scores['tnr'] = tnr
    scores['ppv'] = ppv
    scores['f_score'] = f_score
    scores['ap'] = ap
    scores['brier'] = brier
    scores['recall'] = recall
    
    return scores