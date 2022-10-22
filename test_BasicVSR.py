from __future__ import division
import os
import argparse
from config import get_test_config

parser = argparse.ArgumentParser(description='Test module')
parser.add_argument('--model', type=str, default='model_BasicVSR', help='base model')
parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--scale', type=int, default=4, help='Multiples of super resolution, default:4X')
parser.add_argument('--save_image', type=bool, default=False)
args = parser.parse_args()
opt = get_test_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
from torch.utils.data import DataLoader
from models_BasicVSR.model_BasicVSR import BasicVSRNet
import utils
from data.my_datasets_BasicVSR import myData
from tqdm import tqdm
from models.spatial_color_alignment import color_correction

# load model
net = BasicVSRNet(scale=opt.scale, spynet_pretrained=None)
net = net.cuda()

# load train/test datasets
test_dataloader = DataLoader(myData(opt, 'test'), batch_size=opt.batch_size,
                             shuffle=False, drop_last=False, num_workers=opt.n_workers)

# Load best Network weight
weights = torch.load(os.path.join(opt.weight_savepath, '{0:s}_{1:d}X/best.pth'.format(opt.model, opt.scale)))
net.load_state_dict(weights['state_dict'])
print('Weight loading succeeds')

net.eval()
rgb_psnr_list = []
rgb_ssim_list = []
rgb_cor_psnr_list = []
rgb_cor_ssim_list = []

for test_data in tqdm(test_dataloader):
    with torch.no_grad():
        LRs_RGB = test_data['LRs_RGB'].cuda()
        HRs_RGB_gt = test_data['HRs_RGB'].cuda()
        HRs_RGB_cor = torch.zeros(HRs_RGB_gt.size()).cuda()

        HRs_RGB = net(LRs_RGB)

        for i in range(0, HRs_RGB_gt.shape[1]):
            HRs_RGB_cor[:, i] = color_correction(HRs_RGB_gt[:, i], LRs_RGB[:, i], HRs_RGB[:, i], opt.scale)

            if opt.save_image:
                utils.save_RGB(HRs_RGB_cor[:, i], str(opt.scale), test_data['RGB_gt_name'][i], 'BasicVSR')

            # To calculate average PSNR, SSIM
            rgb_psnr_list.append(utils.get_psnr(HRs_RGB_gt[:, i], HRs_RGB[:, i]))
            rgb_ssim_list.append(utils.get_ssim(HRs_RGB_gt[:, i], HRs_RGB[:, i]))
            rgb_cor_psnr_list.append(utils.get_psnr(HRs_RGB_gt[:, i], HRs_RGB_cor[:, i]))
            rgb_cor_ssim_list.append(utils.get_ssim(HRs_RGB_gt[:, i], HRs_RGB_cor[:, i]))

rgb_psnr = sum(rgb_psnr_list) / len(rgb_psnr_list)
rgb_ssim = sum(rgb_ssim_list) / len(rgb_ssim_list)
rgb_cor_psnr = sum(rgb_cor_psnr_list) / len(rgb_cor_psnr_list)
rgb_cor_ssim = sum(rgb_cor_ssim_list) / len(rgb_cor_ssim_list)

print('valid_rgb_psnr:{:.2f} valid_rgb_ssim:{:.4f} valid_rgb_cor_psnr:{:.2f} valid_rgb_cor_ssim:{:.4f}'
      .format(rgb_psnr, rgb_ssim, rgb_cor_psnr, rgb_cor_ssim))
