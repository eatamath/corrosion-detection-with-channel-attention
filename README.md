# metallic

### 项目结构说明

./

preprocess.ipynb --- for image preprocessing

./dataset --- our dataset path

​				/train --- path of training dataset (`ImageFolder` in Pytorch)

​				/test --- path of test dataset

./utility --- helper module



### metrics.py introduction

about the computation of metrics (defined by many evaluation metrics)

```python
computeMetrics(Ypred:list,Ytest:list) -> dict: {'acc':acc...}
```



### output.py usage for raw image preprocessing

In this step, the raw image will be fragmented in a sliding window manner, and the label of each patch are computed by the overlap with annotated bbox (annotated by LabelMe), i.e., foreground and different levels of corrosion.

- three arguments for `DataProcessor` initialization
  - file path of images and annotations
  - folder of background images
  - folder of foreground images
- four arguments for `process` method for image fragmentation (sliding window)
  - sliding window length
  - sliding window width
  - sliding window x-axis step size
  - sliding window y-axis step size



###### 文件结构

test放的都是测试文件，垃圾内容

utility就是所有有用的模块

reference放大家自己的参考资料

model放模型结果

process放注释内容

log_res放日志文件

###### PS

大家有任何代码上的疑问可随时提问

##### [2020-5-3 11:12 zs commit to master]

###### 命令行参数

增加了argparser后可以使用命令行来执行，具体参数见`add_argument`

###### 参数保存

增加了参数保存方法（未测试）

###### 云文件管理

可以用 `process/upload.ipynb` 直接操作云端文件，具体方法在  `process/readme.md`

这些文件都能共享协作，比较方便

###### 模型参数

模型初步参数已经调整好了，在 `preprocess.ipynb` 中，可以直接运行




