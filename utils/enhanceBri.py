import cv2
import numpy as np
import matplotlib.pyplot as plt


# 直方图均衡增强
def hist(image):
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo


# 限制对比度自适应直方图均衡化CLAHE
def clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe


# 对数变换
def log(image):
    image_log = np.uint8(np.log(np.array(image) + 1))
    cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
    # 转换成8bit图像显示
    cv2.convertScaleAbs(image_log, image_log)
    return image_log


if __name__ == '__main__':
    path = r'E:\cloudremove\PSTCR\s2-33_color\result\result_0.tif'
    image = cv2.imread(path)
    image = clahe(clahe(image))
    # image_equal_clo = hist(image)
    out = r'C:\Users\53110\Desktop\paper\real\PSTCR\20180915test.tif'
    cv2.imwrite(out, image)
    # plt.savefig(path + '输出图片.svg', format='svg', dpi=150)  # 输出
