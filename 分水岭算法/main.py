import cv2
import numpy as np

# 加载图像
img = cv2.imread('lena.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 阈值分割，将图像分为黑白两部分
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow("thresh", thresh)

# 先腐蚀再膨胀
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# cv2.imshow("opening", opening)

# 对3结果进行膨胀，得到大部分都是背景的区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# cv2.imshow("sure_bg", sure_bg)

# 通过distanceTransform获取前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

# cv2.imshow("sure_fg", sure_fg)

# sure_bg与sure_fg相减,得到既有前景又有背景的重合区域   #此区域和轮廓区域的关系未知
sure_fg = np.uint8(sure_fg)
unknow = cv2.subtract(sure_bg, sure_fg)

# 连通区域处理
ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号  序号为 0 - N-1
markers = markers + 1
# 分水岭算法对物体做的标注必须都 大于1 ，背景为标号 为0  因此对所有markers 加1  变成了  1  -  N
# 去掉属于背景区域的部分（即让其变为0，成为背景）
markers[unknow == 255] = 0

# 分水岭算法
markers = cv2.watershed(img, markers)  # 分水岭算法后，所有轮廓的像素点被标注为  -1
# print(markers)

img[markers == -1] = [0, 0, 255]  # 标注为-1 的像素点标 红
cv2.imshow("dst", img)
cv2.imwrite("result.jpg", img)
cv2.waitKey(0)
