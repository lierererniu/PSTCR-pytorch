import numpy as np
import cv2
import os
from PIL import Image
from attrdict import AttrMap
import yaml
from utils import read_simple_tif, mkdir

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
item_width = 256
item_height = 256

if __name__ == '__main__':
    with open('../config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    # maskpath = config.mask_dir + 'mask_cloud_20180925.tif'
    namelist = os.listdir(r'H:\Sentinel2_Dataset\S2_Data\Mask')
    low_index = 0.05
    high_index = 0.25
    length = 12 / 4
    result = '../data/mask' + '_' + str(int(100 * low_index)) + '#' + str(int(100 * high_index)) + \
             '#' + str(int(item_height))
    # mkdir(result)
    # index = 0
    # full_ = item_height * item_width
    # strides = [0, 9, 13, 21, 29, 33, 44, 55, 6, 19, 58, 89, 45]
    # for stride in strides:
    #     for name in namelist:
    #         print(name)
    #         maskpath = r'H:\Sentinel2_Dataset\S2_Data\Mask\\' + name
    #         data = read_simple_tif(maskpath)
    #         data = data[stride:, stride:]
    #         height, width = data.shape
    #         for i in range(0, int(height / item_height)):
    #             for j in range(0, int(width / item_width)):
    #                 cropped = data[i * item_height:(i + 1) * item_height, j * item_width:(j + 1) * item_width]
    #                 # 判断云量
    #                 if int(full_ * high_index) >= np.where(cropped == 0)[0].size >= int(full_ * low_index):
    #                     cv2.imwrite(result + '/' + str(index) + '.tif', cropped)
    #                     index = index + 1

    index = 0
    mask_ = '../dataset/test/55'

    mkdir(mask_)
    for i in range(int(length)):
        for maskname in os.listdir(result):
            if maskname[0:-4] == str(i + 12):
                mask = (1 - (read_simple_tif(result + '/' + maskname) / 255)) * 255
                cv2.imwrite(mask_ + '/' + str(index) + '.bmp', mask)
                cv2.imwrite(mask_ + '/' + str(index + int(length)) + '.bmp', mask)
                cv2.imwrite(mask_ + '/' + str(index + int(length * 2)) + '.bmp', mask)
                cv2.imwrite(mask_ + '/' + str(index + int(length * 3)) + '.bmp', mask)
                index = index + 1


    # print(index)
    # pnglist = os.listdir(r'H:\RICE_DATASET\RICE2\mask')
    # mask_ = r'H:\RFR-Inpainting-master\datasets\RICE2\testing_mask_dataset'
    # for name in pnglist:
    #         print(name)
    #         maskpath = r'H:\RICE_DATASET\RICE2\mask\\' + name
    #         data = np.array(Image.open(maskpath))
    #         data[data >= 128] = 255
    #         data = data[:, :, 0]
    #         cv2.imwrite(mask_ + '//' + name, data)
