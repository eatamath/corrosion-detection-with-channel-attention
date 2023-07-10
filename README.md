# metallic

This is repo for our paper: [A channel attention based deep neural network for automatic metallic corrosion detection](https://www.sciencedirect.com/science/article/abs/pii/S2352710221009049).

### project structure specification

preprocess.ipynb --- for image preprocessing

./dataset --- our dataset path

​ /train --- path of training dataset (ImageFolder in Pytorch)

​ /test --- path of test dataset

./utility --- helper module


### metrics.py introduction

about the computation of metrics (defined by many evaluation metrics)

```python
computeMetrics(Ypred:list, Ytest:list) -> {'acc':acc...}
```


### output.py usage for raw image preprocessing

In this step, the raw image will be fragmented in a sliding window manner, and the label of each patch are computed by the overlap with annotated bbox (annotated by LabelMe), i.e., foreground and different levels of corrosion.

three arguments for DataProcessor initialization
- file path of images and annotations
- folder of background images
- folder of foreground images

four arguments for process method for image fragmentation (sliding window)
- sliding window length
- sliding window width
- sliding window x-axis step size
- sliding window y-axis step size
