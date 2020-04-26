# metallic

### 项目结构说明

./

preprocess.ipynb --- 图片预处理+迁移模型的初步训练

./dataset --- 数据集

​				/train --- 训练数据

​				/test --- 测试数据

./utility

​				/\__init\__.py

​				/output.py --- 文件输出模块



### metrics.py介绍

关于metrics的计算方式

```python
computeMetrics(Ypred:list,Ytest:list) -> {'acc':acc...}
```



### output.py使用方式

- 初始化DataProcessor类时传入三个参数
  - 图片和标注的文件路径
  - 背景图片存放路径
  - 前景图片存放路径
- 调用process( ) 方法进行图片分割需依次传入四个参数
  - sliding window的长
  - sliding window的宽
  - sliding window x方向移动的步长
  - sliding window y方向移动的步长

#### eg

```python
crop = DataProcessor("./crop/", "dataset2/0/", "dataset2/1/")
crop.process(128, 128, 100, 100)
```



存放图片文件目录需如下所示

![image-20200423180449499](/Users/luyumin/Library/Application Support/typora-user-images/image-20200423180449499.png)



### 变更日志

##### [2020-4-26 12:15 zs commit to master]

###### 模型持久化

```python
#### 模型保存
def checkpoint(net:PyTorch-Model,model='se_resnet20'# model-name # ): None
#### 模型恢复
def modelrestore(model='se_resnet20' # model-name #): PyTorch-Model
```

###### Global变量

```python
#### 模型保存文件路径
MODEL_SAVE_PATH = './model'
#### 数据路径
DATA_PATH = r'E:\buffer\dataset\train'
```

###### Image Transformer

```python
#### image transformation
data_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
])
```

可以修改，但是要说明原因

###### 目前的模型

```python
hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet20',
    num_classes=NUM_CLASS,
)
net = hub_model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=P_lr)
```

