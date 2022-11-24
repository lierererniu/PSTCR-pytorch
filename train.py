import os
import time
import albumentations as A
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import yaml
from attrdict import AttrMap
from torch import nn
from torch.utils.data import DataLoader
import pytorch_ssim
from modules.BasicBlock import TwoStepLoss, oneStepLoss
from modules.PSTCR import Spatio_temporal
from pytorch_msssim import MS_SSIM
from utils.data_torch import TrainDataset, tDataset
from utils.lossReport import lossReport, TestReport
from utils.utils import gpu_manage, checkpoint, adjust_learning_rate, print_model, set_lr
from val import val
from torchsummary import summary
ssim_loss = pytorch_ssim.SSIM()
restLoss = TwoStepLoss()
preloss = oneStepLoss()
MSSSIM = MS_SSIM(data_range=1., size_average=True, channel=1).cuda()


trfm = A.Compose([
    A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.SmallestMaxSize(max_size=90, interpolation=1, always_apply=False, p=0.5),
    # A.RandomScale(scale_limit=0.5, interpolation=1, always_apply=False, p=0.5),
    # A.CenterCrop(40, 40, p=1)
])


def Train(config):
    gpu_manage(config)
    print('===> Loading datasets')
    dataset = TrainDataset(config, trfm)
    print('dataset:', len(dataset))
    train_size = int((1 - config.validation_size) * len(dataset))
    validation_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, validation_size])
    print('train dataset:', len(train_ds))
    print('validation dataset:', len(val_ds))
    training_data_loader = DataLoader(dataset=train_ds, num_workers=config.threads, batch_size=config.batchsize,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_ds, num_workers=config.threads,
                                        batch_size=config.validation_batchsize, shuffle=False)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # PSTCR
    model = Spatio_temporal(120, 60, (1, 1), 1, (3, 3))
    criterionMSE = nn.MSELoss()
    if config.cuda:
        model = model.cuda()
        criterionMSE.cuda()
    paras = set_lr(model, config)
    print(paras)

    summary(model, [(1, 256, 256), (4, 256, 256)])
    optimizer = torch.optim.Adam(params=paras,
                                 lr=config.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)

    logreport = lossReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)
    print('===> begin')
    start_time = time.time()
    best_ssim = -1
    best_psnr = -1
    psnr = 0
    ssim = 0
    for epoch in range(config.epoch):
        epoch_start_time = time.time()
        adjust_learning_rate(optimizer, epoch, config.lr, 20)
        print_model(optimizer)
        losses = np.ones((len(training_data_loader)))
        for batch_idx, data in enumerate(training_data_loader):
            gt, Temp_patch, Mask_patch = data[0].cuda(), data[1].cuda(), data[2].cuda()
            cloud = gt * Mask_patch

            # PSTCR
            outputs = model(cloud, Temp_patch)
            loss = restLoss(gt, outputs, Mask_patch)
            # loss = criterionMSE(output, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses[batch_idx] = loss.item()
            print(
                "===> Epoch[{}]({}/{}): loss: {:.4f} ".format(
                    epoch, batch_idx, len(training_data_loader), loss.item()))
        log = {}
        log['epoch'] = epoch

        log['loss'] = np.average(losses)
        logreport(log)
        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        with torch.no_grad():
            log_validation, psnr, ssim = val(config, validation_data_loader, model, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        if epoch % config.snapshot_interval == 0 or (epoch + 1) == config.epoch:
            checkpoint(config, epoch, model)
        if epoch % 1 == 0:
            if psnr >= best_psnr and ssim >= best_ssim:
                print("epoch{}, max_psnr:{:.4f} dB, max_ssim:{:.4f}".format(epoch, psnr, ssim))
                torch.save(model.state_dict(), os.path.join(config.out_dir + '/models', 'best.pth'))
                best_psnr = psnr
                best_ssim = ssim
        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)
    if config.is_train:
        Train(config)
