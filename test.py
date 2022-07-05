from __future__ import division
import os
import argparse
from config import get_test_config

parser = argparse.ArgumentParser(description='Test module')
parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--scale', type=int, default=3, help='Multiples of super resolution, default:4X')
parser.add_argument('--save_image', type=bool, default=False)
args = parser.parse_args()
opt = get_test_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
from torch.utils.data import DataLoader
from models.model import RRVSR
import utils
from data.my_datasets import myData
from tqdm import tqdm
from models.spatial_color_alignment import color_correction

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RRVSR(nf=64, nframes=opt.N_frames, scale=opt.scale)
net = net.to(device)

# load train/test datasets
test_dataloader = DataLoader(myData(opt, 'test'), batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.n_workers)

# Load best Network weight
weights = torch.load(os.path.join(opt.weight_savepath, '{0:s}_{1:d}X/best.pth'.format(opt.model, opt.scale)))
net.load_state_dict(weights['state_dict'])
print('Weight loading succeeds')

net.eval()
rgb_cor_psnr_list = []
rgb_cor_ssim_list = []

for test_data in tqdm(test_dataloader):

    with torch.no_grad():

        LRs_RAW = test_data['LRs_RAW'].cuda()
        LRs_RAW_nopack = test_data['LRs_RAW_nopack'].cuda()
        HR_RGB_gt = test_data['HR_RGB'].cuda()
        LR_RGB = test_data['LR_RGB'].cuda()
        
        HR_RGB = net(LRs_RAW_nopack, LRs_RAW)
        HR_RGB_cor = color_correction(HR_RGB_gt, LR_RGB, HR_RGB, opt.scale)

        rgb_cor_psnr_list.append(utils.get_psnr(HR_RGB_gt, HR_RGB_cor))
        rgb_cor_ssim_list.append(utils.get_ssim(HR_RGB_gt, HR_RGB_cor))


        # Save image
        if opt.save_image:

            # utils.save_RGB(HR_RGB_gt, str(opt.scale), test_data['RGB_gt_name'], 'GT')
            utils.save_RGB(HR_RGB_cor, str(opt.scale), test_data['RGB_gt_name'], 'ours')

test_rgb_cor_psnr = sum(rgb_cor_psnr_list) / len(rgb_cor_psnr_list)
test_rgb_cor_ssim = sum(rgb_cor_ssim_list) / len(rgb_cor_ssim_list)

print('test_rgb_cor_psnr:{:.2f} test_rgb_cor_ssim:{:.4f}'.format(test_rgb_cor_psnr, test_rgb_cor_ssim))