import numpy as np
from torch.autograd import Variable
import pytorch_ssim

ssim_loss = pytorch_ssim.SSIM()


def val(config, test_data_loader, model, criterionMSE, epoch):
    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    model.eval()
    for i, data in enumerate(test_data_loader):
        gt, Temp_patch, Mask_patch = Variable(data[0]), Variable(data[1]), Variable(data[2])
        if config.cuda:
            gt = gt.cuda()
            Temp_patch = Temp_patch.cuda()
            Mask_patch = Mask_patch.cuda()
        # if epoch % config.snapshot_interval == 0:
        #     chanel = 1
        #     w = 1
        cloud = gt * Mask_patch
        output = model(cloud, Temp_patch)
        mse = criterionMSE(output, gt)
        # 1是指max_val
        psnr = 10 * np.log10(1 / mse.item())

        # img1 = output.cpu().numpy()[0, :3].transpose(1, 2, 0)
        # img2 = gt.cpu().numpy()[0, :3].transpose(1, 2, 0)
        ssim = ssim_loss(output, gt)
        avg_mse += mse.item()
        avg_psnr += psnr
        avg_ssim += ssim
    avg_mse = avg_mse / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    print("===> Avg. MSE: {:.4f}".format(np.sqrt(avg_mse)))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} ".format(avg_ssim))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['mse'] = avg_mse
    log_test['psnr'] = avg_psnr
    # log_test['ssim'] = avg_ssim

    return log_test, avg_psnr, avg_ssim
