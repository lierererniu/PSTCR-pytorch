import os
import sys
import cv2
from osgeo import gdal
import numpy as np
from scipy.signal import convolve2d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from utils.lossReport import TestReport

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def Cc_Value(Cloud_1, All_Temp):
    n, m = Cloud_1.shape
    temp1 = 0
    temp2 = 0
    temp3 = 0
    Cloud_1_mean = np.mean(Cloud_1)
    All_Temp_mean = np.mean(All_Temp)
    for i in range(0, n):
        for j in range(0, m):
            temp1 = (Cloud_1[i][j] - Cloud_1_mean) * (All_Temp[i][j] - All_Temp_mean) + temp1
            temp2 = (Cloud_1[i][j] - Cloud_1_mean) * (Cloud_1[i][j] - Cloud_1_mean) + temp2
            temp3 = (All_Temp[i][j] - All_Temp_mean) * (All_Temp[i][j] - All_Temp_mean) + temp3

    if temp2 * temp3 != 0:
        cc = temp1 / np.sqrt(temp2 * temp3)
    else:
        cc = -2

    return cc


def Temporal_Linear_Fit2(Cloud_1, Temp_2, Mask, Mask_temp):
    clean_num = np.where(Mask == 1)[0].size
    x = np.zeros([clean_num])
    y = np.zeros([clean_num])
    w, h = Cloud_1.shape

    # search non-cloud data 搜索无云区
    i = 0
    for j in range(0, w):
        for k in range(0, h):
            if Mask[j][k] != 0 and Mask_temp[j][k]:
                y[i] = Cloud_1[j][k]
                x[i] = Temp_2[j][k]
                i = i + 1

    parameter = np.polyfit(x, y, 1)

    return parameter[0] * Temp_2 + parameter[1]
    # return parameter


def Evaluation_index(A, B):
    n, m = B.shape
    temp1 = 0.0
    temp2 = 0.0
    temp3 = 0.0
    A_mean = np.mean(A)
    B_mean = np.mean(B)
    for i in range(0, n):
        for j in range(0, m):
            temp1 = (A[i][j] - A_mean) * (B[i][j] - B_mean) + temp1
            temp2 = (A[i][j] - A_mean) * (A[i][j] - A_mean) + temp2
            temp3 = (B[i][j] - B_mean) * (B[i][j] - B_mean) + temp3

    cc_cur = temp1 / np.sqrt(temp2 * temp3)

    rmse_cur = np.sqrt(mean_squared_error(B, A))
    return cc_cur, rmse_cur




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
    return data


if __name__ == '__main__':
    validationreport = TestReport(log_dir='./WLR//Evaluation//', log_name='tc20220109')
    resultPath = r'D:\去云对比方法\WLR\tc\result0109\result'
    gtpath = r'D:\cloudromove\Xian\256\20220109'
    log_test = {}
    avg_rmse = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_cc = 0
    for name in os.listdir(resultPath):
        path1 = os.path.join(gtpath, name[7:])
        path2 = os.path.join(resultPath, name)
        gt = read_simple_tif(path1) / 255
        result = read_simple_tif(path2) / 255
        PSNR = peak_signal_noise_ratio(gt, result)
        cc_cur, rmse_cur = Evaluation_index(gt, result)
        ssim = structural_similarity(gt, result)
        log_test['img'] = name
        log_test['rmse'] = rmse_cur
        log_test['psnr'] = PSNR
        log_test['ssim'] = ssim
        log_test['cc'] = cc_cur
        validationreport(log_test)
        log_test = {}
        avg_psnr += PSNR
        avg_ssim += ssim
        avg_cc += cc_cur
        avg_rmse += rmse_cur

    avg_cc = avg_cc / len(os.listdir(resultPath))
    avg_rmse = avg_rmse / len(os.listdir(resultPath))
    avg_ssim = avg_ssim / len(os.listdir(resultPath))
    avg_psnr = avg_psnr / len(os.listdir(resultPath))
    print("average accuracy: psnr:{:.4f} dB, ssim:{:.4f}, cc:{:.4f}, rmse:{:.5f}".format(avg_psnr, avg_ssim, avg_cc,
                                                                                         avg_rmse))

    # img2 = r'D:\去云对比方法\WLR\Test Data\215.tif'
    # img1 = r'D:\去云对比方法\WLR\20211121\215.tif'
    # # img2 = r'D:\cloudromove\dataset\test\20211203\290.tif'
    # # img2 = r'D:\cloudromove\dataset\test\20211211\140.tif'
    # # img1 = r'E:\PSTCR\result\伪彩色\3.TIF'
    # # img2 = r'E:\PSTCR\result\伪彩色\gt3.TIF'
    # im1 = read_simple_tif(img1) / 255
    # im2 = read_simple_tif(img2) / 255
    # # im1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    # # im2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    #
    # PSNR = peak_signal_noise_ratio(im1, im2)
    # cc_cur, rmse_cur = Evaluation_index(im1, im2)
    # ssim = structural_similarity(im1, im2)
    # print(PSNR)
    # # print(ssim1)
    # print(ssim)
    # # im_color1 = cv2.applyColorMap(im1, cv2.COLORMAP_JET)
    # # cv2.imwrite(r'./result/reconstruction/1.png', im_color1)
    # # im_color2 = cv2.applyColorMap(im2, cv2.COLORMAP_JET)
    # # cv2.imwrite(r'./result/reconstruction/2.png', im_color2)
