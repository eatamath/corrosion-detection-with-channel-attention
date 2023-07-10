import os
import json
from PIL import Image
from shapely.geometry import Polygon  # 多边形


# data1 = [(0, 0), (0, 1), (1, 1), (1, 0)]  # 带比较的第一个物体的顶点坐标
# data2 = [(0.4, 0.6), (0.5, 3), (2, 0.5), (2, 2), (2, 1)]  # 待比较的第二个物体的顶点坐标

def judgeType(data1, data2, data3, slideSize, labelFlag):
    """
        任意两个图形的相交面积的计算
        :param data1: 当前物体
        :param data2: 待比较的物体
        :param data3: 待比较的物体(label)
        :return: 当前物体与待比较的物体的面积交集
        """
    flag = 2
    poly1 = Polygon(data1).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(data2).convex_hull
    poly3 = Polygon(data3).convex_hull

    if not poly1.intersects(poly2):
        inter_area1 = 0  # 如果两四边形不相交
    else:
        inter_area1 = poly1.intersection(poly2).area  # 相交面积
    inter_percent = inter_area1 / slideSize

    # print(inter_percent)

    if inter_percent <= 0.05:
        flag = 0 #bg
    else:
        flag = 1
    # if inter_percent >= 1:
    #     flag = 1 #fg
    
    if not poly1.intersects(poly3): #包含标签
        pass
    else:
        inter_area2 = poly1.intersection(poly3).area
        inter_percent = inter_area2 / inter_area1

        # print(inter_percent)

        if inter_percent >= 0.2:
            flag = 0 #bg
            # print('label')
        else:
            pass
            
    return flag

def doCropAndSave(img, desPath, fgPoints, labelPoints, xSize, ySize, xStrip, yStrip, fileName, labelFlag):
    ftype = [(""), ('LX','TX'), ('LY','TY'), ('lx','ly','tx','ty'), ('nx','ny','mx','my')]
    slideSize = xSize * ySize #移动窗口面积
    imgXSize = img.size[0] #图片尺寸
    imgYSize = img.size[1]
    nowXPos = 0
    nowYPos = 0
    count = 1
    while nowYPos + ySize <= imgYSize:
        while nowXPos + xSize <= imgXSize:

            
            flag = 0 # 0:bg 1:fg
            # slidePoints = [(nowXPos, nowYPos), (nowXPos + xSize, nowYPos), (nowXPos, nowYPos + ySize), (nowXPos + xSize, nowYPos + ySize)]
            slidePoints = [(nowXPos, nowYPos), (nowXPos + xSize, nowYPos), (nowXPos + xSize, nowYPos + ySize), (nowXPos, nowYPos + ySize)]
            # slidePoints.append(slidePoints[0])
            # print(slidePoints, labelPoints, fgPoints)

            flag = judgeType(slidePoints, fgPoints, labelPoints, slideSize, labelFlag)

            slideImg = img.crop((slidePoints[0][0], slidePoints[0][1], slidePoints[2][0], slidePoints[2][1]))
            if flag == 0: #bg
                savePosition = os.path.join(desPath, '0')
                # print(savePosition)
            elif flag == 1: #fg
                saveType = ''
                prefix = fileName[0: 2] #获取图片名前缀 eg. lx ly ...
                for i, tp in enumerate(ftype):
                    if prefix in tp:
                        saveType = str(i)
                        break
                if saveType=='':
                    raise Exception('prefix err')
                # print(saveType)
                savePosition = os.path.join(desPath, saveType)
            else:
                count += 1
                nowXPos += xStrip
                continue
            

            if not os.path.exists(savePosition):
                os.mkdir(savePosition)

            saveName = fileName.split('.')[0] + '-$' + str(count) + '-$' + str(nowXPos) + '-$' + str(nowYPos) + '.jpg'
            slideImg.save(os.path.join(savePosition, saveName))
            # with open('./test.log', 'a+') as f:
            # print(f'{saveName} {savePosition}')
                # f.write(f'{saveName} {savePosition}')
            count += 1

            nowXPos += xStrip
        nowXPos = 0
        nowYPos += yStrip



def cropPic(srcPath, desPath, xSize, ySize, xStrip, yStrip):
    '''

    :param srcPath: 图片根目录 eg /srcPath/pic1.jpg ......
    :param desPath: 存储根目录 despath目录结构为 .desPath/0, .desPath/1, .desPath/2, .desPath/3, .desPath/4,
    :param xSize: slidingWindow x方向长度
    :param ySize: slidingWindow y方向长度
    :param xStrip: slidingWindow x方向步长
    :param yStrip: slidingWindow y方向长度
    :return: 无
    '''
    dirs = os.listdir(srcPath)
    print(dirs)
    for pic in dirs:
        print('process', pic)
        prefix, suffix = pic.split(".")
        if suffix == 'json':
            continue
        jsonFile = ''
        try:
            with open(os.path.join(srcPath, prefix + '.json') , 'r') as f:
                # jsonFile = f.read()
                jsonResult = json.load(f)
            img = Image.open(os.path.join(srcPath, pic))
            labelFlag = 0
            fgPoints = jsonResult["shapes"][0]['points']
            # fgPoints.append(fgPoints[0])
            labelPoints = []
            try:
                labelPoints = jsonResult["shapes"][1]['points']
                labelFlag = 0
                # print('lbox', labelPoints)

                # labelPoints.append(labelPoints[0])
            except Exception as e:
                # print('nolabel')
                pass
            # labelPoints =
            doCropAndSave(img, desPath, fgPoints, labelPoints, xSize, ySize, xStrip, yStrip, pic, labelFlag)
        except Exception as e:
            print(pic, e)


# -----------------> X
# |
# |
# |
# |
# |
# \/
# Y
#图片名称
# tx12-n-7-512-256.jpg
# 原始名称(tx12-n)-该图片的第n个slidingwindow-该slidingwindow左上角x坐标-该slidingwindow左上角y坐标
if __name__ == '__main__':
    patch_sz = 64
    src_dir1 = r'E:\Dataset\metallic-research\1\1\Corrosion'
    src_dir2 = r'E:\Dataset\metallic-research\1\1\Non_Corrosion'
    src_dir3 = r'E:\Dataset\metallic-research\2\2'
    tar_dir = os.path.join(r'E:\Dataset\metallic-research', str(patch_sz))
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)

    cropPic(src_dir1, tar_dir, patch_sz, patch_sz, patch_sz, patch_sz)
    cropPic(src_dir2, tar_dir, patch_sz, patch_sz, patch_sz, patch_sz)
    cropPic(src_dir3, tar_dir, patch_sz, patch_sz, patch_sz, patch_sz)