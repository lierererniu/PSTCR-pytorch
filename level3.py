import math
import sys
import time
import numpy as np
import torch
from PIL import Image
from osgeo import gdal
from Evaluation_Index import Cc_Value, Temporal_Linear_Fit2
from modules.PSTCR import Spatio_temporal
from utils.utils import mkdir
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as SSIM
from Evaluation_Index import Evaluation_index
from utils.lossReport import TestReport


def save_bmp(img, out):
    outputImg = Image.fromarray(img * 255)
    outputImg = outputImg.convert('P')
    outputImg.save(out)


def predict(cloud_patch, Temp_patch, Mask_patch, path, use_cuda):
    model = Spatio_temporal(120, 60, (1, 1), 1, (3, 3))
    device = torch.device('cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    width = cloud_patch.shape[0]
    height = cloud_patch.shape[1]
    Cloud_patch = np.zeros([1, 1, width, height])
    temp_patch = np.zeros([1, 4, width, height])
    mask_patch = np.zeros([1, 1, width, height])
    Cloud_patch[0, 0, :, :] = cloud_patch
    for i in range(4):
        temp_patch[0, i, :, :] = Temp_patch[:, :, i]
    mask_patch[0, 0, :, :] = Mask_patch

    Cloud_patch = torch.from_numpy(Cloud_patch.astype(np.float32))
    temp_patch = torch.from_numpy(temp_patch.astype(np.float32))
    mask_patch = torch.from_numpy(mask_patch.astype(np.float32))

    if use_cuda:
        cloud_patch, Temp_patch, Mask_patch = Cloud_patch.to(device), temp_patch.to(device), mask_patch.to(device)
    with torch.no_grad():
        cloud_patch, Temp_patch, Mask_patch = torch.autograd.Variable(cloud_patch), \
                                              torch.autograd.Variable(Temp_patch), torch.autograd.Variable(Mask_patch)
        outputs = model(cloud_patch, Temp_patch)

    return outputs.data.cpu().detach().numpy()


def read_simple_tif(inpath):
    """
    读取少量变量
    :param inpath:栅格数据的输入路径
    :return: 栅格数组，列，行
    """
    ds = gdal.Open(inpath)
    # 判断是否读取到数据
    if ds is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出

    col = ds.RasterXSize
    row = ds.RasterYSize
    dt = ds.GetRasterBand(1)
    data = dt.ReadAsArray()

    del ds
    return data / 255


def PSTCR(gtlist, is_real, modelpath, log_name, out_dir, outpath):
    validationreport = TestReport(log_dir=r'E:\cloudremove\testresult\Evaluation', log_name=log_name)
    avg_rmse = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_cc = 0
    path = modelpath
    log_test = {}
    for name in gtlist:
        Temp_1 = read_simple_tif(r'D:\cloudromove\real\sentinel-2\20180816_\\' + name)
        Temp_2 = read_simple_tif(r'D:\cloudromove\real\sentinel-2\20180905_\\' + name)
        Temp_3 = read_simple_tif(r'D:\cloudromove\real\sentinel-2\20180915_\\' + name)
        Temp_4 = read_simple_tif(r'D:\cloudromove\real\sentinel-2\20180925_\\' + name)
        Temp_5 = read_simple_tif(r'D:\cloudromove\real\sentinel-2\20181020_\\' + name)
        if is_real == True:
            bh = '.tif'
        else:
            bh = '.bmp'
        ori_mask_1 = (
                1 - cv2.imread(r'D:\cloudromove\real\sentinel-2\20180816_mask_\\' + name[:-4] + bh,
                               cv2.IMREAD_GRAYSCALE) / 255)
        ori_mask_2 = (
                1 - cv2.imread(r'D:\cloudromove\real\sentinel-2\20180905_mask_\\' + name[:-4] + bh,
                               cv2.IMREAD_GRAYSCALE) / 255)
        ori_mask_3 = (
                1 - cv2.imread(r'D:\cloudromove\real\sentinel-2\20180915_mask_\\' + name[:-4] + bh,
                               cv2.IMREAD_GRAYSCALE) / 255)
        ori_mask_4 = (
                1 - cv2.imread(r'D:\cloudromove\real\sentinel-2\20180925_mask_\\' + name[:-4] + bh,
                               cv2.IMREAD_GRAYSCALE) / 255)
        ori_mask_5 = (
                1 - cv2.imread(r'D:\cloudromove\real\sentinel-2\20181020_mask_\\' + name[:-4] + bh,
                               cv2.IMREAD_GRAYSCALE) / 255)
        ori_mask_1_1 = ori_mask_1
        ori_mask_2_1 = ori_mask_2
        ori_mask_3_1 = ori_mask_3
        ori_mask_4_1 = ori_mask_4
        ori_mask_5_1 = ori_mask_5

        # 五张雾图叠加
        Cloud_1 = Temp_1 * ori_mask_1
        Original_1 = Cloud_1
        Cloud_2 = Temp_2 * ori_mask_2
        Original_2 = Cloud_2
        Cloud_3 = Temp_3 * ori_mask_3
        Original_3 = Cloud_3
        Cloud_4 = Temp_4 * ori_mask_4
        Original_4 = Cloud_4
        Cloud_5 = Temp_5 * ori_mask_5
        Original_5 = Cloud_5
        w, h = Cloud_1.shape
        # 分配空间
        res_1 = np.zeros([w, h])
        res_2 = np.zeros([w, h])
        res_3 = np.zeros([w, h])
        res_4 = np.zeros([w, h])
        res_5 = np.zeros([w, h])

        W = np.zeros([w, h])
        Cloud_iter_1 = np.zeros([w, h])
        Cloud_iter_2 = np.zeros([w, h])
        Cloud_iter_3 = np.zeros([w, h])
        Cloud_iter_4 = np.zeros([w, h])
        Cloud_iter_5 = np.zeros([w, h])

        Cloud_final_1 = np.zeros([w, h])
        Cloud_final_2 = np.zeros([w, h])
        Cloud_final_3 = np.zeros([w, h])
        Cloud_final_4 = np.zeros([w, h])
        Cloud_final_5 = np.zeros([w, h])

        Mask_final_1 = np.zeros([w, h])
        Mask_final_2 = np.zeros([w, h])
        Mask_final_3 = np.zeros([w, h])
        Mask_final_4 = np.zeros([w, h])
        Mask_final_5 = np.zeros([w, h])

        patch = 500
        ratio = 0.1
        stride = 90
        iter_1 = 0
        last_rest_radio_1 = 0
        All_Temp = np.zeros([patch, patch, 4])
        ALL_Mask = np.zeros([patch, patch, 4])
        Temp_patch = np.zeros([patch, patch, 4])
        CC_List = []
        # 存在云，继续迭代
        while np.where(ori_mask_3 == 0)[0].size != 0:
            iter_1 = iter_1 + 1
            # 迭代一次

            time_start = time.time()
            for x in range(0, w - patch + 1, stride):
                for y in range(0, w - patch + 1, stride):

                    # 无云像素
                    intact_numbers = np.where(ori_mask_3[x: x + patch, y: y + patch] == 1)[0].size

                    # 无需去云
                    # if intact_numbers == patch * patch:
                    if intact_numbers == patch * patch or intact_numbers < patch * patch * ratio:
                        continue
                    else:  # 需要填充
                        All_Temp[:, :, 0] = Original_1[x: x + patch, y: y + patch]
                        All_Temp[:, :, 1] = Original_2[x: x + patch, y: y + patch]
                        All_Temp[:, :, 2] = Original_4[x: x + patch, y: y + patch]
                        All_Temp[:, :, 3] = Original_5[x: x + patch, y: y + patch]

                        ALL_Mask[:, :, 0] = ori_mask_1_1[x: x + patch, y: y + patch]
                        ALL_Mask[:, :, 1] = ori_mask_2_1[x: x + patch, y: y + patch]
                        ALL_Mask[:, :, 2] = ori_mask_4_1[x: x + patch, y: y + patch]
                        ALL_Mask[:, :, 3] = ori_mask_5_1[x: x + patch, y: y + patch]
                        Cc_temp_1 = Cc_Value(Cloud_3[x: x + patch, y: y + patch], All_Temp[:, :, 0]
                                             * ori_mask_3[x: x + patch, y: y + patch])
                        Cc_temp_2 = Cc_Value(Cloud_3[x: x + patch, y: y + patch], All_Temp[:, :, 1]
                                             * ori_mask_3[x: x + patch, y: y + patch])
                        Cc_temp_3 = Cc_Value(Cloud_3[x: x + patch, y: y + patch], All_Temp[:, :, 2]
                                             * ori_mask_3[x: x + patch, y: y + patch])
                        Cc_temp_4 = Cc_Value(Cloud_3[x: x + patch, y: y + patch], All_Temp[:, :, 3]
                                             * ori_mask_3[x: x + patch, y: y + patch])
                        # Cc_temp_1 = np.where(ALL_Mask[:, :, 0] == 1)[0].size
                        # Cc_temp_2 = np.where(ALL_Mask[:, :, 1] == 1)[0].size
                        # Cc_temp_3 = np.where(ALL_Mask[:, :, 2] == 1)[0].size
                        # Cc_temp_4 = np.where(ALL_Mask[:, :, 3] == 1)[0].size
                        CC_List = [Cc_temp_1, Cc_temp_2, Cc_temp_3, Cc_temp_4]

                        # 从小到大列表z
                        cc_max = sorted(CC_List)
                        # 元素索引序列
                        pos_max = sorted(range(len(CC_List)), key=lambda k: CC_List[k], reverse=False)

                        parameter = Temporal_Linear_Fit2(Cloud_3[x: x + patch, y: y + patch],
                                                         All_Temp[:, :, pos_max[3]],
                                                         ori_mask_3[x: x + patch, y: y + patch],
                                                         ALL_Mask[:, :, pos_max[3]])

                        # 输入网络的图1
                        Cloud_patch = Cloud_3[x: x + patch, y: y + patch]
                        _ = Cloud_3[x: x + patch, y: y + patch]
                        # #####

                        mask_patch = 1 - ori_mask_3[x: x + patch, y: y + patch]

                        # 输入网络的图
                        Temp_patch[:, :, 0] = parameter
                        Temp_patch[:, :, 1] = parameter
                        Temp_patch[:, :, 2] = parameter
                        Temp_patch[:, :, 3] = parameter
                        # ###
                        # 纹理恢复网络

                        output = predict(Cloud_patch, Temp_patch, mask_patch, path, use_cuda=True)
                        output = output[0, 0, :, :]
                        # output = MinMaxStander(output)

                        Res_patch = output * mask_patch + _ * ori_mask_3[x: x + patch, y: y + patch]

                        # Update Cloud image and Weight
                        patch_weight = math.exp(1 / (patch * patch - intact_numbers))
                        Cloud_iter_1[x: x + patch, y: y + patch] += Res_patch * patch_weight
                        W[x: x + patch, y: y + patch] = W[x: x + patch, y: y + patch] + patch_weight
            print('Update final image and mask of current iteration:{}'.format(iter_1))
            # Update final image and mask of current iteration
            for i in range(0, w):
                for j in range(0, h):
                    if W[i][j] == 0:
                        Cloud_final_1[i][j] = Cloud_3[i][j]
                        Mask_final_1[i][j] = ori_mask_3[i][j]
                    else:
                        Cloud_final_1[i][j] = Cloud_iter_1[i][j] / W[i][j]
                        Mask_final_1[i][j] = 1
            Cloud_3 = Cloud_final_1
            ori_mask_3 = Mask_final_1

            Cloud_iter_1 = Cloud_iter_1 * 0
            W = W * 0
            Cloud_final_1 = Cloud_final_1 * 0
            Mask_final_1 = Mask_final_1 * 0
            rest_ratio = 100 * np.where(ori_mask_3 == 0)[0].size / (w * h)
            if rest_ratio == last_rest_radio_1:
                stride = stride - 1

            if stride < 1:
                stride = 1
            last_rest_radio_1 = rest_ratio
            time_end = time.time()
            t_time = time_end - time_start
            print("Iteration: {}, rest of missing Regions = {:.3f}%, takes {} seconds. ".format(iter_1, rest_ratio,
                                                                                                round(t_time, 4)))


        ssim = SSIM(Temp_3, Cloud_3)
        psnr = peak_signal_noise_ratio(Temp_3, Cloud_3)
        cc, rmse = Evaluation_index(Cloud_3, Temp_3)

        print(name, ": ssim:{:.4f}, psnr:{:.4f} dB, cc:{:.4f}, rmse:{:.7f}".format(ssim, psnr, cc, rmse))
        if not os.path.exists(out_dir + outpath):
            mkdir(out_dir + outpath + '/' + 'result')
            mkdir(out_dir + outpath + '/' + 'gt')
            mkdir(out_dir + outpath + '/' + 'cloud')

        save_bmp(Cloud_3, out_dir + outpath + '/' + 'result' + '/result_' + name)
        save_bmp(Temp_3, out_dir + outpath + '/' + 'gt' + '/gt_' + name)
        save_bmp(Original_3, out_dir + outpath + '/' + 'cloud' + '/cloud_' + name)

        avg_psnr += psnr
        avg_ssim += ssim
        avg_cc += cc
        avg_rmse += rmse
        log_test['img'] = name
        log_test['rmse'] = rmse
        log_test['psnr'] = psnr
        log_test['ssim'] = ssim
        log_test['cc'] = cc
        validationreport(log_test)
        log_test = {}
        print(name)
    avg_cc = avg_cc / len(gtlist)
    avg_rmse = avg_rmse / len(gtlist)
    avg_ssim = avg_ssim / len(gtlist)
    avg_psnr = avg_psnr / len(gtlist)
    print("average accuracy: psnr:{:.4f} dB, ssim:{:.4f}, cc:{:.4f}, rmse:{:.6f}".format(avg_psnr, avg_ssim, avg_cc,
                                                                                         avg_rmse))


if __name__ == '__main__':
    # gtlist = os.listdir(r'C:\Users\53110\Desktop\农田\20211211')
    # gtlist = ['23.tif', '31.tif', '43.tif', '71.tif', '79.tif', '91.tif', '119.tif', '127.tif',
    #           '139.tif', '167.tif', '175.tif', '187.tif']
    gtlist = ['0.tif', '1.tif', '2.tif']
    gtlist.sort(key=lambda x: int(x[:-4]))
    masklist = os.listdir(r'D:\cloudromove\real\sentinel-2\20180915_mask_')
    masklist.sort(key=lambda x: int(x[:-4]))
    modelpath = r'E:\cloudremove\result\PSTCR\models\best.pth'
    outdir = r'E:\cloudremove\PSTCR\\'
    outpath = 's2-33'
    start_time = time.time()
    PSTCR(gtlist, True, modelpath, log_name='s2-33', out_dir=outdir, outpath=outpath)
    print('avg test time:', (time.time() - start_time) / len(gtlist))