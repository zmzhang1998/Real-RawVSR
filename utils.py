import os
import cv2
import torch
import torchvision.utils as torchutils
import torch.nn.functional as F
import math
import numpy as np
from pytorch_msssim import ssim

def save_checkpoints(state, is_best, save_dir):
    """Saves checkpoint to disk"""
    path = os.path.dirname(save_dir) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(state, save_dir)
    if is_best:
        torch.save(state, path + 'best.pth')


def save_feamap(image, scale, save_name, model_name, i):

    path = './results/fea_map/' + scale + 'X/' + model_name
    if not os.path.exists(path):
        os.makedirs(path)

    image = image.cpu().numpy()
    batch_num = len(image)

    for idx in range(batch_num):

        image = (image[idx] - np.min(image[idx])) / (np.max(image[idx]) - np.min(image[idx]))
        cv2.imwrite(path + os.path.basename(save_name[idx]).replace('.png', '_{:d}.jpg'.format(i)), np.uint8(image*255))

def save_RGB_LR(image, scale, save_name, model_name):

    savepath = './results/RGB/' + scale + 'X/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    batch_num = len(image)

    for idx in range(batch_num):
        torchutils.save_image(image[idx], savepath + os.path.basename(save_name[idx]).replace('LR', 'SR').replace('.png', '_{:s}.png'.format(model_name)))

def save_RGB(image, scale, save_name, model_name):

    savepath = './results/RGB/' + scale + 'X/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    batch_num = len(image)

    for idx in range(batch_num):
        torchutils.save_image(image[idx], savepath + os.path.basename(save_name[idx]).replace('HR', 'SR').replace('.png', '_{:s}.png'.format(model_name)))
        # torchutils.save_image(image[idx], savepath + os.path.basename(save_name[idx]))

def save_RGB_RawVD(image, scale, save_name, model_name):

    savepath = './results/RawVD_diff/' + scale + 'X/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    batch_num = len(image)

    for idx in range(batch_num):
        torchutils.save_image(image[idx], savepath + os.path.basename(save_name[idx]).replace('.png', '_{:s}.png'.format(model_name)))

def get_loss(out_im, gt_im, mask=None):
    if mask is None:
        return torch.abs(out_im - gt_im).mean()
    else:
        return torch.abs((out_im - gt_im) * mask).mean()

def get_CharbonnierLoss(out_im, gt_im, valid=None):
    if valid is None:
        diff = out_im - gt_im
        loss = torch.sqrt(diff * diff + 1e-6).mean()
        return loss
    else:
        diff = out_im - gt_im
        loss = torch.sqrt(diff * diff + 1e-6) * valid
        return loss.mean()

def get_mseloss(out_im, gt_im, valid=None):
    if valid is None:
        return F.mse_loss(out_im, gt_im)
    else:
        return F.mse_loss(out_im * valid, gt_im * valid)

# def get_psnr(HR_gt, HR):
#     HR_gt = HR_gt.detach().clone()
#     HR = HR.detach().clone()
#     diff = (HR - HR_gt).pow(2).mean() + 1e-8
#     psnr = -10 * math.log10(diff)
#     return psnr

def get_ssim(HR_gt, HR):
    HR_gt = HR_gt.detach().clone()
    HR = HR.detach().clone()
    ssim_all = ssim(HR_gt, HR, data_range=1, size_average=True)
    return ssim_all

def get_psnr(HR_gt, HR):
    mse = F.mse_loss(HR_gt, HR, reduction='none')

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    psnr = 10.0 * math.log10(intensity_max / torch.mean(mse))
    return psnr

def pack_rggb_raw(raw):
    # pack RGGB Bayer raw to 4 channels
    _, _, H, W = raw.shape
    raw_pack = torch.cat((raw[:, :, 0:H:2, 0:W:2],
                          raw[:, :, 0:H:2, 1:W:2],
                          raw[:, :, 1:H:2, 0:W:2],
                          raw[:, :, 1:H:2, 1:W:2]), dim=1).cuda()
    return raw_pack

def depack_rggb_raw(raw):
    # depack 4 channels raw to RGGB Bayer
    _, H, W = raw.shape
    output = np.zeros((H * 2, W * 2))

    output[0:2 * H:2, 0:2 * W:2] = raw[0, :, :]
    output[0:2 * H:2, 1:2 * W:2] = raw[1, :, :]
    output[1:2 * H:2, 0:2 * W:2] = raw[2, :, :]
    output[1:2 * H:2, 1:2 * W:2] = raw[3, :, :]

    return output
