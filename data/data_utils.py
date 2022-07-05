from __future__ import division

import numpy as np
import cv2
import random
import torch

def read_img(path, israw=True, user_black=2047, user_sat=16200):

    if israw:
        
        img = cv2.imread(path, -1).astype(np.float32)

        img = np.maximum(img - user_black, 0) / (user_sat - user_black)
        img = np.clip(img, 0, 1)
        img = np.expand_dims(img, axis=0)

    else:

        img = cv2.imread(path, -1).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # w, h = img.size
        img = img / 255.
        img = np.clip(img, 0, 1).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
    return img


def random_crop(img_LRs_RAW_list, img_LR_RGB, img_HR_RGB, LR_size, scale):
    # random crop LR from origin to 64, HR from origin to 256 (size)
    _, H, W = img_LRs_RAW_list[0].shape

    rnd_w = random.randint(0, W - LR_size) // 2 * 2
    rnd_h = random.randint(0, H - LR_size) // 2 * 2

    img_LRs_RAW_list = [v[:, rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size] for v in img_LRs_RAW_list]
    img_LR_RGB = img_LR_RGB[:, rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size]

    rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
    img_HR_RGB = img_HR_RGB[:, rnd_h_HR:rnd_h_HR + LR_size * scale, rnd_w_HR:rnd_w_HR + LR_size * scale]
    return img_LRs_RAW_list, img_LR_RGB, img_HR_RGB


def pack_rggb_raw(raw):
    # pack RGGB Bayer raw to 4 channels
    _, H, W = raw.shape
    raw_pack = np.concatenate((raw[:, 0:H:2, 0:W:2],
                               raw[:, 0:H:2, 1:W:2],
                               raw[:, 1:H:2, 0:W:2],
                               raw[:, 1:H:2, 1:W:2]), axis=0)
    return raw_pack


def pack_rggb_raws(raws):
    # pack RGGB Bayer raw to 4 channels
    _, _, H, W = raws.shape
    raws_pack = np.concatenate((raws[:, :, 0:H:2, 0:W:2],
                                raws[:, :, 0:H:2, 1:W:2],
                                raws[:, :, 1:H:2, 0:W:2],
                                raws[:, :, 1:H:2, 1:W:2]), axis=1)
    return raws_pack


def depack_rggb_raws(raws):
    # depack 4 channels raw to RGGB Bayer
    N, C, H, W = raws.shape
    output = torch.zeros((N, 1, H * 2, W * 2))

    output[:, :, 0:2 * H:2, 0:2 * W:2] = raws[:, 0:1, :, :]
    output[:, :, 0:2 * H:2, 1:2 * W:2] = raws[:, 1:2, :, :]
    output[:, :, 1:2 * H:2, 0:2 * W:2] = raws[:, 2:3, :, :]
    output[:, :, 1:2 * H:2, 1:2 * W:2] = raws[:, 3:4, :, :]

    return output
