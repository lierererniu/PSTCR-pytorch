import os
import cv2
import random
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.backends import cudnn
import sys
from osgeo import gdal
from PIL import Image


def gpu_manage(config):
    if config.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_ids))
        config.gpu_ids = list(range(len(config.gpu_ids)))

    print(os.environ['CUDA_VISIBLE_DEVICES'])

    if config.manualSeed == 0:
        config.manualSeed = random.randint(1, 10000)

    print('Random Seed: ', config.manualSeed)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    if config.cuda:
        torch.cuda.manual_seed_all(config.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def save_image(out_dir, x, num, epoch, filename=None):
    test_dir = os.path.join(out_dir, 'epoch_{0:04d}'.format(epoch))
    if filename is not None:
        test_path = os.path.join(test_dir, filename)
    else:
        test_path = os.path.join(test_dir, 'test_{0:04d}.png'.format(num))

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    cv2.imwrite(test_path, x)


def save_bmp(img, out):
    outputImg = Image.fromarray(img * 255)
    outputImg = outputImg.convert('P')
    outputImg.save(out)


def checkpoint(config, epoch, model):
    model_dir = os.path.join(config.out_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_out_path = os.path.join(model_dir, 'model_epoch_{}.pth'.format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_dir))


def MinMaxStander(data, reverse=True):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if not reverse:
        if len(data.shape) <= 2:
            data = MinMaxScaler().fit_transform(data)

        else:
            for i in range(data.shape[2]):
                data[:, :, i] = MinMaxScaler().fit_transform(data[:, :, i])
    else:
        if len(data.shape) <= 2:
            data = scaler.inverse_transform(data)
        else:
            for i in range(data.shape[2]):
                data[:, :, i] = scaler.inverse_transform(data[:, :, i])
    return data


def read_simple_tif(inpath):
    """
    :param inpath:???????????????????????????
    :return: ????????????????????????
    """
    ds = gdal.Open(inpath)
    # ???????????????????????????
    if ds is None:
        print('Unable to open *.tif')
        sys.exit(1)  # ??????
    dt = ds.GetRasterBand(1)
    data = dt.ReadAsArray()
    del ds
    return data


def mkdir(filepath):
    # filepath = path + ':' + '\\' + filename1 + '_' + filename2
    # ????????????????????????
    folder = os.path.exists(filepath)
    # ????????????
    if not folder:
        # ????????????????????????????????????
        os.makedirs(filepath)
        print('-----????????????----- \n')
    else:
        print(filepath + '??????????????? ???\n')


def set_lr(model, config):
    params = []
    conv3_1_param = dict(model.conv3_1.named_parameters())
    for key, value in conv3_1_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': config.lr}]
        else:
            params += [{'params': [value], 'lr': config.lr * 0.1}]

    conv5_1_param = dict(model.conv5_1.named_parameters())
    for key, value in conv5_1_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': config.lr}]
        else:
            params += [{'params': [value], 'lr': config.lr * 0.1}]

    conv7_1_param = dict(model.conv7_1.named_parameters())
    for key, value in conv7_1_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': config.lr}]
        else:
            params += [{'params': [value], 'lr': config.lr * 0.1}]
    conv3_2_param = dict(model.conv3_2.named_parameters())
    for key, value in conv3_2_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': config.lr}]
        else:
            params += [{'params': [value], 'lr': config.lr * 0.1}]

    conv5_2_param = dict(model.conv5_2.named_parameters())
    for key, value in conv5_2_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': config.lr}]
        else:
            params += [{'params': [value], 'lr': config.lr * 0.1}]

    conv7_2_param = dict(model.conv7_2.named_parameters())
    for key, value in conv7_2_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': config.lr}]
        else:
            params += [{'params': [value], 'lr': config.lr * 0.1}]

    params += [{'params': model.baseConv.parameters(), 'lr': config.lr}]
    params += [{'params': model.conv10.parameters(), 'lr': config.lr}]
    conv11_param = dict(model.conv11.named_parameters())
    for key, value in conv11_param.items():
        if 'bias' not in key:
            params += [{'params': [value], 'lr': config.lr}]
        else:
            params += [{'params': [value], 'lr': 0}]
    return params


def adjust_learning_rate(optimizer, epoch, lr, step):
    """
    ??????lr ???20???epoch????????????
    :param optimizer: ?????????
    :param epoch: ????????????
    :param lr: ?????????
    :return: None
    """
    lr = lr * (0.8 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_model(optimizer):
    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            if k == 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)


class ImageMerge():
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # ?????????????????????????????????
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # ??????????????????
        if len(im_data.shape) == 3:
            # ????????????????????????????????????im_bands, im_height, im_width
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # ??????????????? driver = gdal.GetDriverByName("GTiff")???????????????????????????????????????????????????????????????????????????
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform(im_geotrans)  # ????????????????????????
        dataset.SetProjection(im_proj)  # ????????????

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # ??????????????????
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset