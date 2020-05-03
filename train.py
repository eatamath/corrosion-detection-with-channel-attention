import numpy as np
from scipy.ndimage import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.hub
import torch.utils.data as Data

from torchvision.datasets import *
import torchvision
import torchvision.transforms as transforms

# from torchsummary import summary
# import tensorboardX as tbx
# from tensorboardX import SummaryWriter

import random
import os
import time
import datetime
from PIL import *
import cv2
from cv2 import *

import argparse

from utility.output import *
from utility.metrics import computeMetrics
from utility.network_process import net_freeze_layer
from utility.plot import plotResultCurve
# from utility.edataset import *



    
def getCurrentTime():
    return datetime.datetime.strftime(datetime.datetime.fromtimestamp(time.time()),format='%Y-%m-%d-%H-%M-%S')

#### 模型保存
def checkpoint(model, optimizer, epoch, useTimeDir=False):
    # 保存整个模型  
    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    model_name = str(model).split('(')[0]
    if useTimeDir is True:
        savePath = './'+MODEL_SAVE_PATH+'/'+getCurrentTime()
        os.mkdir(savePath)
    else:
        savePath = MODEL_SAVE_PATH
    dir = os.path.join(savePath,model_name+'_model.pth')
    torch.save(state, dir)
    return savePath if useTimeDir else None

#### 模型恢复
def modelrestore(model):
    model_name = str(model).split('(')[0]
    dir = os.path.join(MODEL_SAVE_PATH,model_name+'_model.pth')
    checkpoint = torch.load(dir)
    model.load_state_dict(checkpoint['net'])
    epoch = checkpoint['epoch'] + 1
    return model, epoch


def saveParameters(root_path):
    params = {
        'EPOCH': EPOCH,
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_CLASS': NUM_CLASS,
        'CV': CV,
        'SCHEDULE_EPOCH': SCHEDULE_EPOCH,
        'SCHEDULE_REGRESS': SCHEDULE_REGRESS,
        '_PARTIAL_TRAIN': _PARTIAL_TRAIN,
        '_PARTIAL_TRAIN_RATIO': _PARTIAL_TRAIN_RATIO,
        '_NET_FREEZE': _NET_FREEZE,
        '_NET_NO_GRAD':  _NET_NO_GRAD,
        'P_lr': P_lr,
        'train_ratio': train_ratio,
    }
    with open(os.path.join(root_path,'params.txt'),'w+') as f:
        json.dump(params,f)
    print('parameters stored')
    return


#### image transformation for original images
data_transform_origin = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])

#### image transformation for augmented images
data_transform_aug = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])


class EDataset(Data.Dataset):

    def __init__(self, root_path, basic_transform, aug_transform, aug_ratio=0.3):
        self.root_path = root_path
        self.basic_transform = basic_transform
        self.aug_transform = aug_transform
        self.image_origin = ImageFolder(self.root_path,
                                        transform = self.basic_transform)
        self.image_augment = ImageFolder(self.root_path,
                                        transform = self.aug_transform)
        self.len_origin = len(self.image_origin)
        self.len_augment = int(len(self.image_augment)*aug_ratio)
        self.idx_augment = np.random.permutation(len(self.image_augment))
        return

    def __len__(self):
        return self.len_origin + self.len_augment

    def __getitem__(self, idx):
        if idx<self.len_origin:
            item = self.image_origin[idx]
        else:
            item = self.image_augment[ self.idx_augment[idx-self.len_origin] ]
        return item

    
def getModel(NUM_CLASS,name='se_resnet50'):
    return torch.hub.load(
            'moskomule/senet.pytorch',
            name,
            num_classes=NUM_CLASS
    )


    
#### 模型保存文件路径
MODEL_SAVE_PATH = './model'
#### 数据路径
DATA_PATH = r'E:\buffer\dataset\train'



if __name__=='__main__':
    try:
        p = argparse.ArgumentParser()

        p.add_argument("--EPOCH", type=int, default=20)
        p.add_argument("--BATCH_SIZE", type=int, default=6)
        p.add_argument("--CV", type=int, default=5)
        p.add_argument("--NUM_CLASS", type=int, default=4)
        p.add_argument("--SCHEDULE_EPOCH", type=int, default=5)
        p.add_argument("--SCHEDULE_REGRESS", type=int, default=0.2)
        p.add_argument("--PARTIAL_TRAIN", action="store_true", default=False)
        p.add_argument("--PARTIAL_TRAIN_RATIO", type=float, default=0.003)
        p.add_argument("--NET_FREEZE", action="store_true", default=False)
        p.add_argument("--train_ratio", type=float, default=0.7)
        p.add_argument("--init_lr", type=float, default=1e-3)

        args = p.parse_args()

        EPOCH = args.EPOCH
        BATCH_SIZE = args.BATCH_SIZE
        NUM_CLASS = args.NUM_CLASS
        CV = args.CV

        SCHEDULE_EPOCH = args.SCHEDULE_EPOCH
        SCHEDULE_REGRESS = args.SCHEDULE_REGRESS

        ### 部分训练
        _PARTIAL_TRAIN = args.PARTIAL_TRAIN
        _PARTIAL_TRAIN_RATIO = args.PARTIAL_TRAIN_RATIO

        ### 冻结网络
        _NET_FREEZE = args.NET_FREEZE
        _NET_NO_GRAD = []

        P_lr = args.init_lr
        train_ratio = args.train_ratio 
    
    except Exception as e:
        print('ERROR:: ',e)
        raise e


    full_dataset = EDataset(DATA_PATH,basic_transform=data_transform_origin,aug_transform=data_transform_aug)
    total_size = len(full_dataset)


    _metrics = []
    epoch_save = 0

    for cv in range(CV):

        hub_model = getModel(NUM_CLASS=NUM_CLASS,name='se_resnet50')

        #### load model
        print('DEBUG:: fold ',cv)
        try:
            hub_model, epo = modelrestore(hub_model)
            print('Model successfully loaded')
            print('-' * 60)
        except Exception as e:
            print('Model not found, use the initial model',e)
            epo = 0
            print('-' * 60)

        net = hub_model

        #### define criterian & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=P_lr)
        scheduler = lr_scheduler.StepLR(optimizer, SCHEDULE_EPOCH, SCHEDULE_REGRESS)

        #### use CUDA if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)


        #### data splitting
        if _PARTIAL_TRAIN:
            full_dataset, _ = torch.utils.data.random_split(full_dataset, 
                                                            [
                                                                int(_PARTIAL_TRAIN_RATIO*total_size),
                                                                total_size - int(_PARTIAL_TRAIN_RATIO*total_size) 
                                                            ])
            total_size = int(_PARTIAL_TRAIN_RATIO*total_size)

        train_size = int(np.floor( total_size * train_ratio ))
        test_size = int(total_size - train_size)
        dataset_train, dataset_test = torch.utils.data.random_split(full_dataset, [train_size,test_size])

        #### training
        _loss = []
        __record_train_num = 0
        epoch_save = epo
        for epoch in range(epo, EPOCH):  # loop over the dataset multiple times
            epoch_save += 1
            print('DEBUG:: training epoch ',epoch)
            trainloader = Data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
            testloader = Data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=True)

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                #### get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                #### zero the parameter gradients
                optimizer.zero_grad()
                #### forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                #### print statistics
                running_loss += loss.item()
                _loss.append(running_loss)

                __record_train_num += len(labels)
                if __record_train_num % (BATCH_SIZE * 50) == 0:
                    print('DEBUG:: num has trained',__record_train_num)
                if i % 50 == 0:
                    print('DEBUG:: trainloader:{}/{}'.format(i, len(trainloader)))
                if i % 50 == 0:
                    try:
                        checkpoint(net, optimizer, epoch_save)
                        print('*' * 60)
                        print('Model is saved successfully at epoch {}'.format(str(epoch)))
                        print('*' * 60)
                    except Exception as e:
                        print('*' * 60)
                        print('Something is wrong!',e)
                        print('*' * 60)

            #### predicting
            print('=' * 60)
            print('i:', i)
            print('Start predicting')
            Ypred = []
            Ytest = []
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, -1)
                    Ytest.extend(labels.tolist())
                    Ypred.extend(predicted.tolist())

            _metrics.append(computeMetrics(Ypred,Ytest))
            print("accuracy is {}".format(_metrics[-1]['acc']) )
            print("auc is {}".format(_metrics[-1]['auc']) )
            print('=' * 60)


    print('-' * 60)
    print('Training is over, saving the model')
    print('-' * 60)
    try:
        savePath = checkpoint(net, optimizer, epoch_save, useTimeDir=True)
        saveResult(_metrics,savePath)
        saveParameters(savePath)
        print('Model is saved successfully')
    except Exception as e:
        print('Something is wrong!',e)
        raise e



