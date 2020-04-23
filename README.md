# 使用方式

- 初始化DataProcessor类时传入三个参数
  - 图片和标注的文件路径
  - 背景图片存放路径
  - 前景图片存放路径
- 调用process( ) 方法进行图片分割需依次传入四个参数
  - sliding window的长
  - sliding window的宽
  - sliding window x方向移动的步长
  - sliding window y方向移动的步长

## eg

```python
crop = DataProcessor("./crop/", "dataset2/0/", "dataset2/1/")
crop.process(128, 128, 100, 100)
```



存放图片文件目录需如下所示

![image-20200423180449499](/Users/luyumin/Library/Application Support/typora-user-images/image-20200423180449499.png)