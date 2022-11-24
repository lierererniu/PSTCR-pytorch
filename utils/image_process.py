from Evaluation_Index import Cc_Value
from utils.utils import read_simple_tif, mkdir
from attrdict import AttrMap
import yaml
import os
import cv2

if __name__ == '__main__':
    with open('../config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)
    length = len(os.listdir(r'../data/20041124_256'))
    years = ['20041124_256', '20041226_256', '20050111_256', '20050127_256', '20050212_256']
    path1 = r'./data/' + years[0]
    path2 = r'./data/' + years[1]
    path3 = r'./data/' + years[2]
    path4 = r'./data/' + years[3]
    path5 = r'./data/' + years[4]
    result = r'E:/PSTCR/data/'
    mask_path = r'../data/mask_20#30#256'

    for n in range(5):
        mkdir(result + years[n] + '_')
    mkdir(result + 'mask_256')
    m = 0
    for i in range(0, length):
        orgin = read_simple_tif(path1 + '/' + str(i) + '.tif')
        imp1 = read_simple_tif(path2 + '/' + str(i) + '.tif')
        imp2 = read_simple_tif(path3 + '/' + str(i) + '.tif')
        imp3 = read_simple_tif(path4 + '/' + str(i) + '.tif')
        imp4 = read_simple_tif(path5 + '/' + str(i) + '.tif')
        mask = read_simple_tif(mask_path + '/' + str(i) + '.tif')
        Cc_temp_1 = Cc_Value(orgin, imp1)
        Cc_temp_2 = Cc_Value(orgin, imp2)
        Cc_temp_3 = Cc_Value(orgin, imp3)
        Cc_temp_4 = Cc_Value(orgin, imp4)
        CC_List = [Cc_temp_1, Cc_temp_2, Cc_temp_3, Cc_temp_4]
        # 从小到大列表
        cc_max = sorted(CC_List)
        # 元素索引序列
        pos_max = sorted(range(len(CC_List)), key=lambda k: CC_List[k], reverse=False)
        index = 0
        for value in cc_max:
            if value >= 0.85:
                index += 1
        if index == 4:
            cv2.imwrite(result + years[0] + '_/' + str(m) + '.tif', orgin)
            cv2.imwrite(result + years[1] + '_/' + str(m) + '.tif', imp1)
            cv2.imwrite(result + years[2] + '_/' + str(m) + '.tif', imp2)
            cv2.imwrite(result + years[3] + '_/' + str(m) + '.tif', imp3)
            cv2.imwrite(result + years[4] + '_/' + str(m) + '.tif', imp4)
            cv2.imwrite(result + 'mask__110/' + str(m) + '.tif', mask)
            m += 1
            print(m)
    print(m)


