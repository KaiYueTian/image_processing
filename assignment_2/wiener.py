# # # import cv2
# # # from PIL import Image
# # # from torchvision.transforms import transforms
# # #
# # # from models.VGG16 import SRM
# # # from PIL import Image
# # #
# # # transformer_data = transforms.Compose([transforms.Resize((224, 224)),
# # #                                        transforms.ToTensor(),
# # #                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# # #                                        ])
# # # frame = Image.open("003.jpg")
# # # frame = transformer_data(frame)
# # # model = SRM()
# # # output = model(frame)
# # # im = output.detach().numpy()
# # # pic = Image.fromarray(im, "RGB")
# # # pic.show()
# # #
# # # print(output.size())
# # # # # #
# # # # import torch
# # # #
# # # # weight1 = (1 / 4) * (torch.Tensor([[[[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0],
# # # #                                      [0, 0, 0, 0, 0]],
# # # #                                     [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0],
# # # #                                      [0, 0, 0, 0, 0]],
# # # #                                     [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0],
# # # #                                      [0, 0, 0, 0, 0]]]]))
# # # # weight2 = (1 / 12) * (torch.Tensor([[[[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
# # # #                                       [-1, 2, -2, 2, -2]],
# # # #                                      [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
# # # #                                       [-1, 2, -2, 2, -2]],
# # # #                                      [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
# # # #                                       [-1, 2, -2, 2, -2]]]]))
# # # # weight3 = (1 / 2) * (torch.Tensor([[[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
# # # #       [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
# # # #       [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]]))
# # # # weight = torch.cat([weight1, weight2, weight3], 0)
# # # # print(weight.size())
# # # # from torch import nn
# # # #
# # # # conv2d = nn.Conv2d(3, 3, kernel_size=5, padding=2)
# # # # print(conv2d.weight.data.size())
# # import cv2, os
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# #
# # def get_img(input_Path):
# #     img_paths = []
# #     for (path, dirs, files) in os.walk(input_Path):
# #         for filename in files:
# #             if filename.endswith(('.jpg', '.png')):
# #                 img_paths.append(path + '/' + filename)
# #     return img_paths
# #
# #
# # # 构建Gabor滤波器
# # def build_filters():
# #     filters = []
# #     ksize = [7, 9, 11, 13, 15, 17]  # gabor尺度，6个
# #     lamda = np.pi / 2.0  # 波长
# #     for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向，0°，45°，90°，135°，共四个
# #         for K in range(6):
# #             kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
# #             kern /= 1.5 * kern.sum()
# #             filters.append(kern)
# #     plt.figure(1)
# #
# #     # 用于绘制滤波器
# #     for temp in range(len(filters)):
# #         plt.subplot(4, 6, temp + 1)
# #         plt.imshow(filters[temp])
# #     plt.show()
# #     return filters
# #
# #
# # # Gabor特征提取
# # def getGabor(img, filters):
# #     res = []  # 滤波结果
# #     for i in range(len(filters)):
# #         # res1 = process(img, filters[i])
# #         accum = np.zeros_like(img)
# #         for kern in filters[i]:
# #             fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
# #             accum = np.maximum(accum, fimg, accum)
# #         res.append(np.asarray(accum))
# #         return res[:, 8]
# #
# #
# # if __name__ == '__main__':
# #     filters = build_filters()
# #     img = cv2.imread("003.jpg")
# # #     getGabor(img, filters)
# # def convert(input_dir, output_dir):
# #     for filename in os.listdir(input_dir):
# #         path = input_dir + "/" + filename # 获取文件路径
# #         print("doing... ", path)
# #         noise_img = cv2.imread(path)#读取图片
# #         img_noise = gaussian_noise(noise_img, 0, 0.12) # 高斯噪声
# #         # img_noise = sp_noise(noise_img,0.025)# 椒盐噪声
# #         #img_noise  = random_noise(noise_img,500)# 随机噪声
# #         cv2.imwrite(output_dir+'/'+filename,img_noise )
# from data_process import DeNoise
#
# train_dir = '/farm/data/FaceForensice++c40image/Face-c40_train.csv'
# # test_dir = "/farm/data/FaceForensice++c40image/Face-c40_test.csv"
# # valid_dir = "/farm/data/FaceForensice++c40image/Face-c40_valid.csv"
# DeNoise(train_dir)
# print("over")
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
import cv2


# 仿真运动模糊
def motion_process(image_size, motion_angle):
    PSF = np.zeros(image_size)
    print(image_size)
    center_position = (image_size[0] - 1) / 2
    print(center_position)

    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


# 对图片进行运动模糊
def make_blurred(input, PSF, eps):
    input_fft = fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = fft.fft2(PSF) + eps
    blurred = fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(fft.fftshift(blurred))
    return blurred


def inverse(input, PSF, eps):  # 逆滤波
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(input_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result


def wiener(input, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result

def wiener_Nk(input, PSF, eps):  # 维纳滤波，K=0.01
    input_fft = fft.fft2(input)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2)
    result = fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result

def normal(array):
    array = np.where(array < 0, 0, array)
    array = np.where(array > 255, 255, array)
    array = array.astype(np.int16)
    return array


def main(gray):
    channel = []
    img_h, img_w = gray.shape[:2]
    PSF = motion_process((img_h, img_w), 60)  # 进行运动模糊处理
    blurred = np.abs(make_blurred(gray, PSF, 1e-3))

    result_blurred = inverse(blurred, PSF, 1e-3)  # 逆滤波
    result_wiener = wiener(blurred, PSF, 1e-3)  # 维纳滤波
    result_wiener_Nk = wiener_Nk(blurred, PSF, 1e-3)  # 维纳滤波

    blurred_noisy = blurred + 0.1 * blurred.std() * \
                    np.random.standard_normal(blurred.shape)  # 添加噪声,standard_normal产生随机的函数
    inverse_mo2no = inverse(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加噪声的图像进行逆滤波
    wiener_mo2no = wiener(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加噪声的图像进行维纳滤波
    wiener_mo2no_Nk = wiener_Nk(blurred_noisy, PSF, 0.1 + 1e-3)  # 对添加噪声的图像进行维纳滤波
    channel.append((normal(blurred), normal(result_blurred), normal(result_wiener), normal(result_wiener_Nk),
                    normal(blurred_noisy), normal(inverse_mo2no), normal(wiener_mo2no), normal(wiener_mo2no_Nk)))
    return channel


if __name__ == '__main__':
    image = cv2.imread('./019.jpg')
    b_gray, g_gray, r_gray = cv2.split(image.copy())

    Result = []
    for gray in [b_gray, g_gray, r_gray]:
        channel = main(gray)
        Result.append(channel)
    blurred = cv2.merge([Result[0][0][0], Result[1][0][0], Result[2][0][0]])
    result_blurred = cv2.merge([Result[0][0][1], Result[1][0][1], Result[2][0][1]])
    result_wiener = cv2.merge([Result[0][0][2], Result[1][0][2], Result[2][0][2]])
    result_wiener_Nk = cv2.merge([Result[0][0][3], Result[1][0][3], Result[2][0][3]])
    blurred_noisy = cv2.merge([Result[0][0][4], Result[1][0][4], Result[2][0][4]])
    inverse_mo2no = cv2.merge([Result[0][0][5], Result[1][0][5], Result[2][0][5]])
    wiener_mo2no = cv2.merge([Result[0][0][6], Result[1][0][6], Result[2][0][6]])
    wiener_mo2no_Nk = cv2.merge([Result[0][0][7], Result[1][0][7], Result[2][0][7]])

    # ========= 可视化 ==========
    plt.figure(1)
    plt.xlabel("Original Image")
    plt.imshow(np.flip(image, axis=2))  # 显示原图像

    plt.figure(2)
    plt.figure(figsize=(8, 6.5))
    imgNames = {"make_blurred": blurred,
                # "inverse deblurred:运动模糊-逆滤波": result_blurred,
                "make_blurred-k=0.01": result_wiener,
                "make_blurred-k=NONE": result_wiener_Nk,
                "make_blurred+random noise": blurred_noisy,
                # "inverse_mo2no": inverse_mo2no,
                'random noise-k=0.01': wiener_mo2no,
                'random noise-k=NONE': wiener_mo2no_Nk}
    for i, (key, imgName) in enumerate(imgNames.items()):
        plt.subplot(231 + i)
        plt.xlabel(key)
        plt.imshow(np.flip(imgName, axis=2))
    plt.show()
