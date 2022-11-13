
import numpy as np
import cv2
from matplotlib import pyplot as plt


def magnitude_phase_split(img):
    # 分离幅度谱与相位谱
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    # 幅度谱
    magnitude_spectrum = np.abs(dft_shift)
    # 相位谱
    phase_spectrum = np.angle(dft_shift)
    return magnitude_spectrum, phase_spectrum

# 读取图像 主图和纹理图
img1 = cv2.imread("lena.jpg", 0)

# 分离幅度谱与相位谱
img_m, img_p = magnitude_phase_split(img1)
# 合并幅度谱与相位谱
img_1 = np.abs(np.fft.ifft2(img_m*np.e**(1j*img_p)))
img_2 = np.abs(np.fft.ifft2(img_m))
img_3 = np.abs(np.fft.ifft2(np.e**(1j*img_p)))


plt.figure(figsize=(10, 10))
plt.subplot(321)
plt.xlabel("img")
plt.imshow(img1, cmap="gray")
plt.subplot(322)
plt.imshow(img_1, cmap="gray")
plt.xlabel("same")
plt.subplot(323)
plt.imshow(img_2, cmap="gray")
plt.xlabel("p = 0")
plt.subplot(324)
plt.imshow(img_3, cmap="gray")
plt.xlabel("f = 1")
plt.show()
