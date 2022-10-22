from __future__ import division
import os
import argparse
from config import get_test_config

parser = argparse.ArgumentParser(description='Test module')
parser.add_argument('--model', type=str, default='model_Bicubic', help='base model')
parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--scale', type=int, default=4, help='Multiples of super resolution, default:4X')
parser.add_argument('--save_image', type=bool, default=False)
args = parser.parse_args()
opt = get_test_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
from data.my_datasets import myData
from tqdm import tqdm
from models.spatial_color_alignment import color_correction

# load train/test datasets
test_dataloader = DataLoader(myData(opt, 'test'), batch_size=opt.batch_size,shuffle=False, drop_last=False, num_workers=opt.n_workers)

# Load best Network weight
rgb_psnr_list = []
rgb_ssim_list = []
rgb_cor_psnr_list = []
rgb_cor_ssim_list = []

for test_data in tqdm(test_dataloader):

    with torch.no_grad():

        LR_RGB = test_data['LR_RGB'].cuda()
        HR_RGB_gt = test_data['HR_RGB'].cuda()

        HR_RGB = F.interpolate(LR_RGB, scale_factor=opt.scale, mode='bicubic', align_corners=False)
        HR_RGB_cor = color_correction(HR_RGB_gt, LR_RGB, HR_RGB, opt.scale)

        rgb_psnr_list.append(utils.get_psnr(HR_RGB_gt, HR_RGB))
        rgb_ssim_list.append(utils.get_ssim(HR_RGB_gt, HR_RGB))
        rgb_cor_psnr_list.append(utils.get_psnr(HR_RGB_gt, HR_RGB_cor))
        rgb_cor_ssim_list.append(utils.get_ssim(HR_RGB_gt, HR_RGB_cor))
        
        # Save image
        if opt.save_image:
            # utils.save_RGB(HR_RGB_gt, str(opt.scale), test_data['RGB_gt_name'], 'GT')
            utils.save_RGB(HR_RGB_cor, str(opt.scale), test_data['RGB_gt_name'], 'Bicubic')

test_rgb_psnr = sum(rgb_psnr_list) / len(rgb_psnr_list)
test_rgb_ssim = sum(rgb_ssim_list) / len(rgb_ssim_list)
test_rgb_cor_psnr = sum(rgb_cor_psnr_list) / len(rgb_cor_psnr_list)
test_rgb_cor_ssim = sum(rgb_cor_ssim_list) / len(rgb_cor_ssim_list)

print('test_rgb_psnr:{0:.2f} test_rgb_ssim:{1:.4f} test_rgb_cor_psnr:{2:.2f} test_rgb_cor_ssim:{3:.4f}'
      .format(test_rgb_psnr, test_rgb_ssim, test_rgb_cor_psnr, test_rgb_cor_ssim))
