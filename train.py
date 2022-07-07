from __future__ import division
import os
import argparse
from config import get_train_config

parser = argparse.ArgumentParser(description='Training module')
parser.add_argument('--gpu_id', type=str, default='1', help='gpu id')
parser.add_argument('--scale', type=int, default=4, help='Multiples of super resolution, default:4X')
parser.add_argument('--continue_train', type=bool, default=False, help='retrain')
args = parser.parse_args()
opt = get_train_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

import torch
torch.manual_seed(1)
torch.cuda.manual_seed(1)

from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from models.model import RRVSR
from models.spatial_color_alignment import color_correction
from tensorboardX import SummaryWriter
import utils
from data.my_datasets import myData

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

log_dir = './logs/train/model/' + opt.model + '_' + str(opt.scale) + 'X'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RRVSR(nf=64, nframes=opt.N_frames, scale=opt.scale)
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=opt.lr)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

# multi-GPU
# device_ids = [Id for Id in range(torch.cuda.device_count())]
# net = torch.nn.DataParallel(net, device_ids=device_ids)

# calculate all trainable parameters in network
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

# continue training
if opt.continue_train:
    net_weights = torch.load(os.path.join(opt.weight_savepath, '{0:s}_{1:d}X/best.pth'.format(opt.model, opt.scale)))

    net.load_state_dict(net_weights['state_dict'])

    print("loading iters %d" % (net_weights['iters']))
    iters = net_weights['iters'] + 1

    print("loading optimizer state_dict")
    optimizer.load_state_dict(net_weights['optimizer'])

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 0.1*1e-4

    print("loading previous_valid_psnr %f" % (net_weights['best_psnr']))
    best_valid_psnr = net_weights['best_psnr']

    rng_state = net_weights['random_state']
    torch.set_rng_state(rng_state)
else:
    best_valid_psnr = 0
    iters = opt.init_iters
    iters_loss = 0

# load train/valid datasets
train_dataloader = DataLoader(myData(opt, 'train'), batch_size=opt.batch_size,
                              shuffle=True, drop_last=False, num_workers=opt.n_workers)

train_val_dataloader = DataLoader(myData(opt, 'train_val'), batch_size=opt.valid_batch_size,
                                  shuffle=True, drop_last=False, num_workers=opt.n_workers)

valid_dataloader = DataLoader(myData(opt, 'valid'), batch_size=opt.valid_batch_size,
                              shuffle=True, drop_last=False, num_workers=opt.n_workers)

net.train()

while iters < opt.num_iters:

    for train_data in train_dataloader:
        iters += 1

        LRs_RAW = train_data['LRs_RAW'].cuda()
        LRs_RAW_nopack = train_data['LRs_RAW_nopack'].cuda()
        HR_RGB_gt = train_data['HR_RGB'].cuda()
        LR_RGB = train_data['LR_RGB'].cuda()

        optimizer.zero_grad()

        HR_RGB = net(LRs_RAW_nopack, LRs_RAW)
        HR_RGB_cor = color_correction(HR_RGB_gt, LR_RGB, HR_RGB, scale_factor=opt.scale)
        loss = utils.get_CharbonnierLoss(HR_RGB_cor, HR_RGB_gt)

        loss.backward()
        optimizer.step()

        iters_loss = loss.item()

        print("\rEpoch:%d iter:%d loss=%.8f" % (iters // len(train_dataloader) + 1, iters, iters_loss), end='')

        if iters % 3000 == 0:

            net.eval()
            rgb_psnr_list = []
            rgb_ssim_list = []
            running_loss = []
            
            for train_val_data in tqdm(train_val_dataloader):
                with torch.no_grad():
                    LRs_RAW = train_val_data['LRs_RAW'].cuda()
                    LRs_RAW_nopack = train_val_data['LRs_RAW_nopack'].cuda()
                    HR_RGB_gt = train_val_data['HR_RGB'].cuda()
                    LR_RGB = train_val_data['LR_RGB'].cuda()

                    HR_RGB = net(LRs_RAW_nopack, LRs_RAW)
                    HR_RGB_cor = color_correction(HR_RGB_gt, LR_RGB, HR_RGB, scale_factor=opt.scale)
                    loss = utils.get_CharbonnierLoss(HR_RGB_cor, HR_RGB_gt)

                    # To calculate average PSNR, SSIM
                    rgb_psnr_list.append(utils.get_psnr(HR_RGB_gt, HR_RGB_cor))
                    rgb_ssim_list.append(utils.get_ssim(HR_RGB_gt, HR_RGB_cor))
                    running_loss.append(loss)

            # Average PSNR on one epoch train_data
            rgb_psnr = sum(rgb_psnr_list) / len(rgb_psnr_list)
            rgb_ssim = sum(rgb_ssim_list) / len(rgb_ssim_list)
            loss = sum(running_loss) / len(running_loss)
            scheduler.step(loss)

            print("iter:%d loss=%.8f rgb_psnr:%.2f rgb_ssim:%.4f"% (iters, loss, rgb_psnr, rgb_ssim))

            writer.add_scalars('PSNR/rgb', {'train_val_rgb_psnr': rgb_psnr}, iters)
            writer.add_scalars('SSIM/rgb', {'train_val_rgb_ssim': rgb_ssim}, iters)
            writer.add_scalars('Loss', {'train_val_loss': loss}, iters)

            vis_image = torch.stack((F.interpolate(LR_RGB[0].unsqueeze(0), scale_factor=opt.scale, mode='bicubic', align_corners=False)[0],
                                     torch.clamp(HR_RGB_cor[0], 0, 1), torch.clamp(HR_RGB_gt[0], 0, 1)))
            writer.add_images('train_val_vis', vis_image, iters, dataformats='NCHW')

            is_best = False
            # use evaluation models during the net evaluating
            lr_check = optimizer.param_groups[0]['lr']
            print('====validation====  lr_check:{}'.format(lr_check))

            rgb_psnr_list = []
            rgb_ssim_list = []
            for valid_data in tqdm(valid_dataloader):
                with torch.no_grad():
                    LRs_RAW = valid_data['LRs_RAW'].cuda()
                    LRs_RAW_nopack = valid_data['LRs_RAW_nopack'].cuda()
                    HR_RGB_gt = valid_data['HR_RGB'].cuda()
                    LR_RGB = valid_data['LR_RGB'].cuda()

                    HR_RGB = net(LRs_RAW_nopack, LRs_RAW)
                    HR_RGB_cor = color_correction(HR_RGB_gt, LR_RGB, HR_RGB, scale_factor=opt.scale)

                    # To calculate average PSNR, SSIM
                    rgb_psnr_list.append(utils.get_psnr(HR_RGB_gt, HR_RGB_cor))
                    rgb_ssim_list.append(utils.get_ssim(HR_RGB_gt, HR_RGB_cor))

            # Average PSNR on one epoch train_data

            rgb_psnr = sum(rgb_psnr_list) / len(rgb_psnr_list)
            rgb_ssim = sum(rgb_ssim_list) / len(rgb_ssim_list)

            print("rgb_psnr:%.2f rgb_ssim:%.4f" % (rgb_psnr, rgb_ssim))

            writer.add_scalars('PSNR/rgb', {'valid_rgb_psnr': rgb_psnr}, iters)
            writer.add_scalars('SSIM/rgb', {'valid_rgb_ssim': rgb_ssim}, iters)

            vis_image = torch.stack((F.interpolate(LR_RGB[0].unsqueeze(0), scale_factor=opt.scale, mode='bicubic', align_corners=False)[0],
                                     torch.clamp(HR_RGB_cor[0], 0, 1), torch.clamp(HR_RGB_gt[0], 0, 1)))
            writer.add_images('valid_random_vis', vis_image, iters, dataformats='NCHW')

            is_best = rgb_psnr >= best_valid_psnr
            if is_best:
                best_valid_psnr = rgb_psnr

            utils.save_checkpoints({'iters': iters,
                                    'state_dict': net.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'random_state': torch.get_rng_state(),
                                    'best_psnr': best_valid_psnr},
                                    is_best,
                                    save_dir=os.path.join(opt.weight_savepath, '{0:s}_{1:d}X/{2:03d}k.pth'.format(opt.model, opt.scale, iters//1000)))

            net.train()
