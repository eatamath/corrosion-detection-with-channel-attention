import torch
from torch.autograd import Variable

import os
import time
import json

def LOG(*kargs):
#     global LOG_LEVEL
    if LOG_LEVEL == 'LOG':
        s = ''
        for arg in kargs:
            s += str(arg)
        s = 'DEBUG:: ' + s + '\n'
        print(s)
    return


def output_netork(net):
    for name, value in net.named_parameters():
        print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
        return

def visualize_network_hist(writer,net,n_iter):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
    return
    
def visualize_network_graph(writer,net,dsize=(256,256)):
    writer.add_graph(net, (Variable(torch.rand(2, 3, dsize[0],dsize[1])),))
    return

def visualize_variable(writer,variable:tuple,n_iter):
    writer.add_scalar('data/'+str(variable[0]), variable[1], n_iter)
    return

def saveResult(_metrics:list,savePath:str):
    with open(os.path.join(savePath,'scores.txt'),'w+') as f:
        json.dump(_metrics,f)
    return
        
def savePredTest(Ypred:list, Ytest:list, savePath:str):
    res = {
        "pred":Ypred,
        "test":Ytest,
    }
    with open(os.path.join(savePath,'pred-test.txt'),'w+') as f:
        json.dump(res,f)
    return