import cv2
import random
import numpy as np
import torch
import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class TrainDataset(data.Dataset):

    def __init__(self, config, transform):
        super().__init__()
        self.config = config
        self.transform = transform
        train_list_file = os.path.join(config.datasets_dir, config.train_list)
        # 如果数据集尚未分割，则进行训练集和测试集的分割
        if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
            files = os.listdir(os.path.join(config.datasets_dir, config.years[0]))
            random.shuffle(files)
            n_train = int(config.train_size * len(files))
            train_list = files[:n_train]
            test_list = files[n_train:]
            np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')
            np.savetxt(os.path.join(config.datasets_dir, config.test_list), np.array(test_list), fmt='%s')
        self.imlist = np.loadtxt(train_list_file, str)

    def __getitem__(self, index):
        cloud = cv2.imread(os.path.join(self.config.datasets_dir, self.config.years[0], str(self.imlist[index])),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x1 = cv2.imread(os.path.join(self.config.datasets_dir, self.config.years[1], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x2 = cv2.imread(os.path.join(self.config.datasets_dir, self.config.years[2], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)

        m = cv2.imread(os.path.join(self.config.mask_dir, self.config.mask_dir_name[0],
                                    (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m1 = cv2.imread(os.path.join(self.config.mask_dir, self.config.mask_dir_name[1],
                                     (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m2 = cv2.imread(os.path.join(self.config.mask_dir, self.config.mask_dir_name[2],
                                     (str(self.imlist[index])[:-4] + '.bmp')), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        m = 1 - (m / 255)
        m1 = (1 - (m1 / 255))
        m2 = (1 - (m2 / 255))

        # 归一化
        temp = np.dstack((cloud, x1, x2)) / 255
        m = np.dstack((m, m1, m2))
        augments = self.transform(image=temp, mask=m)
        gt1 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[0:1, :, :])
        gt2 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[1:2, :, :])
        gt3 = torch.from_numpy(np.transpose(augments['image'], (2, 0, 1))[2:3, :, :])
        gt = [gt1, gt2, gt3]
        mask1 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[0:1, :, :])
        mask2 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[1:2, :, :])
        mask3 = torch.from_numpy(np.transpose(augments['mask'], (2, 0, 1))[2:3, :, :])
        m = [mask1, mask2, mask3]
        return gt, m

    def __len__(self):
        return len(self.imlist)


class tDataset(data.Dataset):

    def __init__(self, config, is_real=False):
        super().__init__()
        self.config = config
        self.is_real = is_real
        test_list_file = os.path.join(config.test_dir, config.test_list)
        if not os.path.exists(test_list_file) or os.path.getsize(test_list_file) == 0:
            files = os.listdir(os.path.join(config.test_dir, config.cloud[0]))
            # np.savetxt(os.path.join(config.test_dir, config.train_list), np.array(files), fmt='%s')
            np.savetxt(os.path.join(config.test_dir, config.test_list), np.array(files), fmt='%s')
        # else:
        #     os.remove(test_list_file)
        #     files = os.listdir(os.path.join(config.test_dir, config.cloud[0]))
        #     # np.savetxt(os.path.join(config.test_dir, config.train_list), np.array(files), fmt='%s')
        #     np.savetxt(os.path.join(config.test_dir, config.test_list), np.array(files), fmt='%s')
        self.imlist = np.loadtxt(test_list_file, str)

    def __getitem__(self, index):
        # [index]
        cloud = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[0], str(self.imlist[index])),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x1 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[1], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        x2 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[2], str(self.imlist[index])),
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # x3 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[3], str(self.imlist[index])),
        #                 cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # x4 = cv2.imread(os.path.join(self.config.test_dir, self.config.cloud[4], str(self.imlist[index])),
        #                 cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if self.is_real:
            end = '.tif'
        else:
            end = '.bmp'
        print(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[0],
                                    (str(self.imlist[index])[:-4] + end)))
        m = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[0],
                                    (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m1 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[1],
                                     (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m2 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[2],
                                     (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # m3 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[3],
        #                              (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # m4 = cv2.imread(os.path.join(self.config.tmask_dir, self.config.mask_dir_name[4],
        #                              (str(self.imlist[index])[:-4] + end)), cv2.IMREAD_GRAYSCALE).astype(np.float32)

        m = 1 - (m / 255)
        m1 = 1 - (m1 / 255)
        m2 = 1 - (m2 / 255)
        # m3 = 1 - (m3 / 255)
        # m4 = 1 - (m4 / 255)
        # 归一化
        temp = np.dstack((cloud, x1, x2)) / 255
        m = np.dstack((m, m1, m2))
        # augments = self.transform(image=temp, mask=m)
        gt1 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[0:1, :, :])
        gt2 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[1:2, :, :])
        gt3 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[2:3, :, :])
        # gt4 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[3:4, :, :])
        # gt5 = torch.from_numpy(np.transpose(temp, (2, 0, 1))[4:5, :, :])
        gt = [gt1, gt2, gt3]
        mask1 = torch.from_numpy(np.transpose(m, (2, 0, 1))[0:1, :, :])
        mask2 = torch.from_numpy(np.transpose(m, (2, 0, 1))[1:2, :, :])
        mask3 = torch.from_numpy(np.transpose(m, (2, 0, 1))[2:3, :, :])
        # mask4 = torch.from_numpy(np.transpose(m, (2, 0, 1))[3:4, :, :])
        # mask5 = torch.from_numpy(np.transpose(m, (2, 0, 1))[4:5, :, :])
        m = [mask1, mask2, mask3]
        name_ = str(self.imlist[index])[:-4] + '.tif'
        if self.is_real:
            name_ = str(self.imlist[index])[:-4] + '.tif'
        return gt, m, name_

    def __len__(self):
        return len(self.imlist)
