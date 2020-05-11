
from PIL import *
import numpy as np
import cv2
from cv2 import *
import os
import random


DATA_ROOT = r'D:\onedrive\eyisheng\OneDrive\files\大三下\科研\金属分类\data'
DATA_ROOT_CORROSION = DATA_ROOT + r'\corrosion'
DATA_ROOT_NONCORROSION = DATA_ROOT + r'\noncorrosion'

DATASET_ROOT = './dataset'
DATASET_ROOT_TRAIN = os.path.join(DATASET_ROOT,'train')
DATASET_ROOT_TEST = os.path.join(DATASET_ROOT,'test')
DATASET_TRAIN_POS = os.path.join(DATASET_ROOT_TRAIN,'1')
DATASET_TRAIN_NEG = os.path.join(DATASET_ROOT_TRAIN,'0')
DATASET_TEST_POS = os.path.join(DATASET_ROOT_TEST,'1')
DATASET_TEST_NEG = os.path.join(DATASET_ROOT_TEST,'0')

if not os.path.exists(DATASET_ROOT_TRAIN):
    os.mkdir(DATASET_ROOT_TRAIN)

if not os.path.exists(DATASET_ROOT_TEST):
    os.mkdir(DATASET_ROOT_TEST)
    
    

#### get raw jpg files from directories
def getFileNames(is_corrosion=False):
    path_files = []
    walk_root = DATA_ROOT_NONCORROSION if is_corrosion==False else DATA_ROOT_CORROSION
    
    for root, dirs, files in os.walk(walk_root):
        for f in files:
            fpath = os.path.join(root,f)
            path_files.append(fpath)
            
    label = 0 if is_corrosion==False else 1
    res = list(map(lambda x:(x,label),path_files))
    return res


def checkImgShape():
    shapes = set()
    for line in non_corrosion_files:
        jpgfile = Image.open(line[0])
        jpgfile = np.array(jpgfile)
        shapes.add(jpgfile.shape)
    return shapes


def resizeImg(src,dsize):
    jpgfile = Image.open(src)
    jpgfile = np.array(jpgfile)
    jpgfile = cv2.resize(jpgfile,dsize)
    return jpgfile


def preprocessingPipeline(src,target):
    img = resizeImg(src,(256,256)) ### image shapes
    return [src,target,img]
    
    
def filesToImg(path_files):
    train = []
    test = []
    for line in path_files:
        if random.random()<0: ### train test split
            test.append( preprocessingPipeline(line[0],line[1]) )
        else:
            train.append( preprocessingPipeline(line[0],line[1]) )
    return {'train':train,'test':test}
    
    
def saveImgToFiles(imgs):
    
    if not os.path.exists(DATASET_TRAIN_POS):
        os.mkdir(DATASET_TRAIN_POS)
    if not os.path.exists(DATASET_TRAIN_NEG):
        os.mkdir(DATASET_TRAIN_NEG) 
    if not os.path.exists(DATASET_TEST_POS):
        os.mkdir(DATASET_TEST_POS)
    if not os.path.exists(DATASET_TEST_NEG):
        os.mkdir(DATASET_TEST_NEG)
        
    for line in imgs['train']:
        src = os.path.split(line[0])[-1]
        if line[1]==0:
            src = os.path.join(DATASET_TRAIN_NEG,src)
        elif line[1]==1:
            src = os.path.join(DATASET_TRAIN_POS,src)
        Image.fromarray(line[2]).save(src)
        
    for line in imgs['test']:
        src = os.path.split(line[0])[-1]
        if line[1]==0:
            src = os.path.join(DATASET_TEST_NEG,src)
        elif line[1]==1:
            src = os.path.join(DATASET_TEST_POS,src)
        Image.fromarray(line[2]).save(src)
    return

def ImgWriter():
    img_data = filesToImg(img_files)

    saveImgToFiles(img_data)
    return

def RemoveImgInFolder(path_dir, ratio) :
    # path文件夹路径，ratio删除比例
    filelist = os.listdir(path_dir)
    rmfilelist = random.sample(filelist, int(ratio*len(filelist)))
    for f in rmfilelist :
        filepath = os.path.join(path_dir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
    return

def ImgSummary(path_dirs):
    filelists = [ os.listdir(x) for x in path_dirs ]
    _ = [print(p,len(x)) for x,p in zip(filelists,path_dirs)]
    return
# jpgfile = Image.open(os.path.join(DATA_PATH,r'2\LY01-N1074.jpg'))
# jpgfile = np.array(jpgfile)
# jpgfile.shape