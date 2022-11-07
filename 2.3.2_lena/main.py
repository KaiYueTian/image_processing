import cv2
from PIL import Image


def transfer(infile, outfile):
    im = Image.open(infile)
    reim = im.resize((144, 192))  # 宽*高

    reim.save(outfile, dpi=(20.0, 20.0))


# 下面函数先量化到level+1级，再量化至256级以显示
def reduce_intensity_levels(img, level):
    img = cv2.copyTo(img, None)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            si = img[x, y]
            ni = int(level * si / 255 + 0.5) * (255 / level)
            img[x, y] = ni
    return img


src = cv2.imread('lena.jpg')

# cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 量化到8级
result8 = reduce_intensity_levels(gray, 7)
# 量化到2级
result2 = reduce_intensity_levels(gray, 1)

cv2.imwrite('result8.jpg', result8)
cv2.imwrite('result2.jpg', result2)
cv2.imwrite('result256.jpg', gray)

infil = r"lena.jpg"
outfile = r"20_lena.jpg"
transfer(infil, outfile)
