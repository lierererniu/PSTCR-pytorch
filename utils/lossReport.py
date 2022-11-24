import json
import os
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt


class lossReport():
    def __init__(self, log_dir, log_name='log'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)

    def save_lossgraph(self):
        epoch = []
        loss = []
        for l in self.log_:
            epoch.append(l['epoch'])
            loss.append(l['loss'])
        epoch = np.asarray(epoch)
        loss = np.asarray(loss)
        plt.plot(epoch, loss)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(self.log_dir, 'loss_graph.png'))
        plt.close()


class TestReport():
    def __init__(self, log_dir, log_name='log_test'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)

    def save_lossgraph(self):
        epoch = []
        mse = []
        psnr = []
        ssim = []
        for l in self.log_:
            epoch.append(l['epoch'])
            mse.append(l['mse'])
            psnr.append(l['psnr'])
            ssim.append(l['ssim'])
        epoch = np.asarray(epoch)
        mse = np.asarray(mse)
        psnr = np.asarray(psnr)
        ssim = np.asarray(ssim)

        plt.plot(epoch, mse)
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.savefig(os.path.join(self.log_dir, 'graph_mse.png'))
        plt.close()

        plt.plot(epoch, psnr)
        plt.xlabel('epoch')
        plt.ylabel('psnr')
        plt.savefig(os.path.join(self.log_dir, 'graph_psnr.png'))
        plt.close()

        plt.plot(epoch, ssim)
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(self.log_dir, 'graph_ssim.png'))
        plt.close()
