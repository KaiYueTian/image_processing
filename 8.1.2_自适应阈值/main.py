import numpy as np
import cv2 as cv


def OTSU(img_gray, GrayScale):
    assert img_gray.ndim == 2, "must input a gary_img"  # shape有几个数字, ndim就是多少
    img_gray = np.array(img_gray).ravel().astype(np.uint8)
    u1 = 0.0  # 背景像素的平均灰度值
    u2 = 0.0  # 前景像素的平均灰度值
    th = 0.0

    # 总的像素数目
    PixSum = img_gray.size
    # 各个灰度值的像素数目
    PixCount = np.zeros(GrayScale)
    # 各灰度值所占总像素数的比例
    PixRate = np.zeros(GrayScale)
    # 统计各个灰度值的像素个数
    for i in range(PixSum):
        # 默认灰度图像的像素值范围为GrayScale
        Pixvalue = img_gray[i]
        PixCount[Pixvalue] = PixCount[Pixvalue] + 1

    # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
    for j in range(GrayScale):
        PixRate[j] = PixCount[j] * 1.0 / PixSum
    Max_var = 0
    # 确定最大类间方差对应的阈值
    for i in range(1, GrayScale):  # 从1开始是为了避免w1为0.
        u1_tem = 0.0
        u2_tem = 0.0
        # 背景像素的比列
        w1 = np.sum(PixRate[:i])
        # 前景像素的比例
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            pass
        else:  # 背景像素的平均灰度值
            for m in range(i):
                u1_tem = u1_tem + PixRate[m] * m
            u1 = u1_tem * 1.0 / w1
            # 前景像素的平均灰度值
            for n in range(i, GrayScale):
                u2_tem = u2_tem + PixRate[n] * n
            u2 = u2_tem / w2
            # print(u1)
            # 类间方差公式：G=w1*w2*(u1-u2)**2
            tem_var = w1 * w2 * np.power((u1 - u2), 2)
            # print(tem_var)
            # 判断当前类间方差是否为最大值。
            if Max_var < tem_var:
                Max_var = tem_var  # 深拷贝，Max_var与tem_var占用不同的内存空间。
                th = i
    return th


def Iterate_Thresh(img, initval, MaxIterTimes=20, thre=1):
    mask1, mask2 = (img > initval), (img <= initval)
    T1 = np.sum(mask1 * img) / np.sum(mask1)
    T2 = np.sum(mask2 * img) / np.sum(mask2)
    T = (T1 + T2) / 2
    # 终止条件
    if abs(T - initval) < thre or MaxIterTimes == 0:
        return T
    return Iterate_Thresh(img, T, MaxIterTimes - 1)

# 读取图片
img = cv.imread('test.jpg', 0)

# 阈值迭代
# 计算灰度平均值
initthre = np.mean(img)
thresh = Iterate_Thresh(img, initthre, 50)
dst_I = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)[1]
cv.imwrite("Iterate.jpg", dst_I)

# 大律法
th = OTSU(img, 256)
dst_O = cv.threshold(img, th, 255, cv.THRESH_BINARY)[1]
cv.imwrite("OTSU.jpg", dst_O)

print("使用Iterate_Thresh的方法：" + str(thresh))
print("使用OTSU的方法：" + str(th))



