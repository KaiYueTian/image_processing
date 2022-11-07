from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np


def show_multi_imgs(scale, imglist, order=None, border=30, border_color=(255, 255, 255)):
    """
    :param scale: float 原图缩放的尺度
    :param imglist: list 待显示的图像序列
    :param order: list or tuple 显示顺序 行×列
    :param border: int 图像间隔距离
    :param border_color: tuple 间隔区域颜色
    :return: 返回拼接好的numpy数组
    """
    if order is None:
        order = [1, len(imglist)]
    allimgs = imglist.copy()
    ws, hs = [], []
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        ws.append(allimgs[i].shape[1])
        hs.append(allimgs[i].shape[0])
    w = max(ws)
    h = max(hs)
    # 将待显示图片拼接起来
    sub = int(order[0] * order[1] - len(imglist))
    # 判断输入的显示格式与待显示图像数量的大小关系
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros(((h + border) * order[0], (w + border) * order[1], 3)) + border_color
    imgblank = imgblank.astype(np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            imgblank[(i * h + i * border):((i + 1) * h + i * border), (j * w + j * border):((j + 1) * w + j * border),
            :] = allimgs[i * order[1] + j]
    return imgblank


gray_lena = "lena_gray.jpg"


def LinearTran(path, new_name, a=1, b=0):
    im = Image.open(path)
    imarray = np.array(im)

    height, width = imarray.shape
    for i in range(height):
        for j in range(width):
            aft = int(a * imarray[i, j] + b)
            if aft <= 255 and aft >= 0:
                imarray[i, j] = aft
            elif aft > 255:
                imarray[i, j] = 255
            else:
                imarray[i, j] = 0
    new_im = Image.fromarray(imarray)
    new_im.save(new_name)


def PiecewiseLinear(path, new_name, a=1, b=0):
    im = Image.open(path)
    imarray = np.array(im)

    height, width = imarray.shape
    for i in range(height):
        for j in range(width):
            aft = int(a * imarray[i, j] + b)
            if aft <= 160 and aft >= 60:
                imarray[i, j] = aft
            elif aft > 160:
                imarray[i, j] = 255
            else:
                imarray[i, j] = 0
    new_im = Image.fromarray(imarray)
    new_im.save(new_name)


def ExpTran(src, new_path, esp=0, gama=1):
    im = Image.open(src)
    imarray = np.array(im)
    height, width = imarray.shape

    for i in range(height):
        for j in range(width):
            tmp = imarray[i, j] / 255
            tmp = int(pow(tmp + esp, gama) * 255)
            if tmp >= 0 and tmp <= 255:
                imarray[i, j] = tmp
            elif tmp > 255:
                imarray[i, j] = 255
            else:
                imarray[i, j] = 0
    new_im = Image.fromarray(imarray)
    new_im.save(new_path)

Linear_path = "Linear.jpg"
LinearTran(gray_lena, Linear_path, a=1, b=55)
PiecewiseLinear_path = "PiecewiseLinear.jpg"
PiecewiseLinear(gray_lena, PiecewiseLinear_path, a=1, b=55)
ExpTran_path = "ExpTran.jpg"
ExpTran(gray_lena, ExpTran_path, 0, 3)

image = cv2.imread(gray_lena)
image1 = cv2.imread(Linear_path)
image2 = cv2.imread(PiecewiseLinear_path)
image3 = cv2.imread(ExpTran_path)
img = show_multi_imgs(0.9, [image, image1, image2, image3], (2, 2))
cv2.namedWindow('multi', 0)
cv2.imshow('multi', img)
cv2.imwrite('result.jpg',img)
cv2.waitKey(0)
