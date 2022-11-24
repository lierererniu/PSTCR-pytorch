import os
import time
from utils import read_simple_tif
import numpy as np
import cv2
import copy
import yaml
from PIL import Image
from attrdict import AttrMap
from utils import mkdir
Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
item_width = 40
item_height = 40


class DataAugment:
    def __init__(self, debug=False):
        self.debug = debug
        print("Data augment...")

    def basic_matrix(self, translation):
        """基础变换矩阵"""
        return np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    def adjust_transform_for_image(self, img, trans_matrix):
        """根据图像调整当前变换矩阵"""
        transform_matrix = copy.deepcopy(trans_matrix)
        height, width = img.shape
        transform_matrix[0:2, 2] *= [width, height]
        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot(
            [self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply_transform(self, img, transform):
        """仿射变换"""
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                                borderValue=0, )  # cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT
        return output

    def apply(self, img, trans_matrix):
        """应用变换"""
        tmp_matrix = self.adjust_transform_for_image(img, trans_matrix)
        out_img = self.apply_transform(img, tmp_matrix)
        if self.debug:
            self.show(out_img)
        return out_img

    def random_vector(self, min, max):
        """生成范围矩阵"""
        min = np.array(min)
        max = np.array(max)
        print(min.shape, max.shape)
        assert min.shape == max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    def show(self, img):
        """可视化"""
        outputImg = Image.fromarray(img)
        # "L"代表将图片转化为灰度图
        outputImg = outputImg.convert('P')
        # outputImg.save('./result/reconstruction/reconstruction_' + str(patch) + '.bmp')
        outputImg.show()

    def random_transform(self, img, min_translation, max_translation):
        """平移变换"""
        factor = self.random_vector(min_translation, max_translation)
        trans_matrix = np.array([[1, 0, factor[0]], [0, 1, factor[1]], [0, 0, 1]])
        out_img = self.apply(img, trans_matrix)
        return trans_matrix, out_img

    def random_flip(self, img, factor):
        """水平或垂直翻转"""
        flip_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, flip_matrix)
        return flip_matrix, out_img

    def random_rotate(self, img, factor, seed):
        """随机旋转"""
        np.random.seed(seed)
        angle = np.random.uniform(factor[0], factor[1])
        print("angle:{}".format(angle))
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        out_img = self.apply(img, rotate_matrix)
        return rotate_matrix, out_img

    def random_scale(self, img, min_translation, max_translation):
        """随机缩放"""
        factor = self.random_vector(min_translation, max_translation)
        scale_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, scale_matrix)
        return scale_matrix, out_img

    def random_shear(self, img, factor):
        """随机剪切，包括横向和众向剪切"""
        angle = np.random.uniform(factor[0], factor[1])
        print("fc:{}".format(angle))
        crop_matrix = np.array([[1, factor[0], 0], [factor[1], 1, 0], [0, 0, 1]])
        out_img = self.apply(img, crop_matrix)
        return crop_matrix, out_img

    def OutResizeImage(self, img, wRate=0.6, hRate=0.75):
        """
        缩小图像比例为（0.6，0.75）
        放大图像比例为（1.5，1.2）
        """
        height, width = img.shape[:2]
        if (wRate < 1) & (hRate < 1):
            # 缩小图像
            size = (int(width * wRate), int(height * hRate))
            shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            if self.debug:
                self.show(shrink)
            return shrink
        else:
            # 放大图像
            fx, fy = wRate, hRate
            enlarge = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            if self.debug:
                self.show(enlarge)
            return enlarge


if __name__ == "__main__":
    with open('../config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)
    length = 14894

    demo = DataAugment(debug=False)
    img = cv2.imread("../data/mask/7.tif", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("../data/20041124/7.tif", cv2.IMREAD_GRAYSCALE)
    outimg = demo.OutResizeImage(img1, wRate=1.2, hRate=1.2)
    # # 平移测试
    # _, outimg = demo.random_transform(img, (0.1, 0.1), (0.2, 0.2))  # (-0.3,-0.3),(0.3,0.3)
    #
    # # 垂直变换测试
    # _, outimg = demo.random_flip(img, (1.0, -1.0))
    #
    # # 水平变换测试
    # _, outimg = demo.random_flip(img, (-1.0, 1.0))

    # 旋转变换测试
    # _, outimg = demo.random_rotate(img, (0.5, 0.8), 10)
    # _, outimg1 = demo.random_rotate(img1, (0.5, 0.8), 10)
    # # 缩放变换测试
    # _, outimg2 = demo.random_scale(img1, (1.2, 1.2), (1.3, 1.3))

    # # 随机裁剪测试
    # _, outimg = demo.random_shear(img, (0.2, 0.3))

    # 组合变换
    # t1, _ = demo.random_transform(img, (-0.3, -0.3), (0.3, 0.3))
    # t2, _ = demo.random_rotate(img, (0.5, 0.8))
    # t3, _ = demo.random_scale(img, (1.5, 1.5), (1.7, 1.7))
    # tmp = np.linalg.multi_dot([t1, t2, t3])
    # print("tmp:{}".format(tmp))
    # out = demo.apply(img, tmp)

    # years = ['20041124', '20041226', '20050111', '20050127', '20050212']

    # #######################################
    # m = 4
    # mask_path = r'../data/mask_40'
    # start_time = time.time()
    # for size in config.size:
    #     m = m + 1
    #     for year in config.years:
    #         path = r'E:\PSTCR\data\\' + year
    #         for k in range(0, length):
    #             mask = read_simple_tif(mask_path + '/' + str(k % length) + '.tif')
    #             data = read_simple_tif(path + '/' + str(k % length) + '.tif')
    #             if year == config.years[0]:
    #                 outmask = demo.OutResizeImage(mask, size, size)
    #                 # _, outmask = demo.random_rotate(mask, (0.5, 0.8), seed)
    #                 cv2.imwrite(mask_path + '/' + str(k + int(length * m)) + '.tif', outmask)
    #             outdata = demo.OutResizeImage(data, size, size)
    #             cv2.imwrite(path + '/' + str(k + int(length * m)) + '.tif', outdata)
    #     print('seed', size, 'finished, use time', time.time() - start_time)
    # print('time:', time.time() - start_time)
    # #######################

    # #######################################
    # m = 0
    # mask_path = r'E:\PSTCR\data\mask_20#30#40'
    # result = r'E:\PSTCR\data\mask_2_3'
    # mkdir(result)
    # start_time = time.time()
    # for seed in config.seeds:
    #     for year in config.years:
    #         m += 1
    #         path = r'E:\PSTCR\data\\' + year
    #         for k in range(0, length):
    #             mask = read_simple_tif(mask_path + '/' + str(k % length) + '.tif')
    #             # data = read_simple_tif(path + '/' + str(k % length) + '.tif')
    #             if year == config.years[0]:
    #                 _, outmask = demo.random_rotate(mask, (0.3, 0.8), seed)
    #                 cv2.imwrite(result + '/' + str(k + int(length * m)) + '.tif', outmask)
    #             # _, outdata = demo.random_rotate(data, (0.3, 0.8), seed)
    #             # cv2.imwrite(path + '/' + str(k + int(length * m)) + '.tif', outdata)
    #     print('seed', seed, 'finished, use time', time.time() - start_time)
    # print('time:', time.time() - start_time)
    # # #######################

    mask_path = r'E:\PSTCR\data\mask_20#30#40'
    result = r'E:\PSTCR\data\mask_2_3'
    mkdir(result)


    length1 = len(os.listdir(mask_path))
    for i in range(7, 8):
        for k in range(0, length1):
            mask = read_simple_tif(mask_path + '/' + str(k % length) + '.tif')
            _, outmask = demo.random_rotate(mask, (0.1, 0.8), (k + length1 * i))

            cv2.imwrite(result + '/' + str(k + length1 * i) + '.tif', outmask)
            print(k)


