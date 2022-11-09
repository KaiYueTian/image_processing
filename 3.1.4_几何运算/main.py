from math import fabs, sin, radians, cos

import cv2
import numpy as np


def rotate_img(img, degrees):
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degrees))) + height * fabs(cos(radians(degrees))))
    widthNew = int(height * fabs(sin(radians(degrees))) + width * fabs(cos(radians(degrees))))

    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    resultImg = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    return resultImg


def pan_img(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    resultImg = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return resultImg


if __name__ == '__main__':
    img = cv2.imread('lena.jpg')
    img_H = cv2.flip(img, 1, dst=None)
    # 图片-垂直镜像
    img_V = cv2.flip(img, 0, dst=None)
    # 图片-对角镜像
    img_HV = cv2.flip(img, -1, dst=None)
    # 平移
    img_pan = pan_img(img, -50, -50)
    # 旋转
    img_rotate = rotate_img(img, 45)

    cv2.imshow('img_H', img_H)
    cv2.imshow('img_V', img_V)
    cv2.imshow('img_HV', img_HV)
    cv2.imshow('img_pan', img_pan)
    cv2.imshow('img_rotate', img_rotate)
    # cv2.imwrite('img_H.jpg', img_H)
    # cv2.imwrite('img_V.jpg', img_V)
    # cv2.imwrite('img_HV.jpg', img_HV)
    # cv2.imwrite('img_pan.jpg', img_pan)
    # cv2.imwrite('img_rotate.jpg', img_rotate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.imread('lena.jpg')
    # 图片-水平镜像
    img_H = cv2.flip(img, 1, dst=None)
    # 旋转
    img_rotate = rotate_img(img_H, 45)
    # 图片-垂直镜像
    img_V = cv2.flip(img_rotate, 0, dst=None)
    # 平移
    img_result = pan_img(img_V, -50, -50)

    cv2.imshow('img_result', img_result)
    # cv2.imwrite('img_result.jpg', img_result)


