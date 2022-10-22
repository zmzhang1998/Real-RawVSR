import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
import data.data_utils as datautils
import glob
import os
import random


class myData(Dataset):

    def __init__(self, opt, mode):
        super(myData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.N_frames = opt.N_frames
        self.half_N_frames = opt.N_frames // 2

        if self.mode == 'train' or self.mode == 'train_val':

            self.paths_HR_RGB = opt.train_paths_HR_RGB
            self.paths_LR_RGB = opt.train_paths_LR_RGB
        else:
            self.paths_HR_RGB = opt.test_paths_HR_RGB
            self.paths_LR_RGB = opt.test_paths_LR_RGB

        self.videos_path_HR_RGB = sorted(glob.glob(os.path.join(self.paths_HR_RGB, '*')))
        self.videos_path_LR_RGB = sorted(glob.glob(os.path.join(self.paths_LR_RGB, '*')))

        if  self.mode == 'train': #采用本文中的做法，即训练时输入为5帧

            self.data_info = {'path_LR_RGB': [], 'path_HR_RGB': [], 'border': []}

            for subfolder_HR_RGB, subfolder_LR_RGB in zip(self.videos_path_HR_RGB,
                                                          self.videos_path_LR_RGB):
                frames_path_HR_RGB = sorted(glob.glob(os.path.join(subfolder_HR_RGB, '*')))
                frames_path_LR_RGB = sorted(glob.glob(os.path.join(subfolder_LR_RGB, '*')))

                assert len(frames_path_HR_RGB) == len(frames_path_LR_RGB), 'Different number of images in LR and HR folders'

                self.data_info['path_HR_RGB'].extend(frames_path_HR_RGB)
                self.data_info['path_LR_RGB'].extend(frames_path_LR_RGB)

                is_border = [0] * len(frames_path_LR_RGB)
                for i in range(self.half_N_frames):
                    is_border[i] = 1
                    is_border[len(frames_path_LR_RGB) - i - 1] = 1
                self.data_info['border'].extend(is_border)

        if  self.mode == 'test': #采用BasicVSR的做法，整个视频序列输入网络

            self.data_info_HR = {}
            self.data_info_LR = {}

            for subfolder_HR_RGB, subfolder_LR_RGB in zip(self.videos_path_HR_RGB,
                                                          self.videos_path_LR_RGB):
                frames_path_HR_RGB = sorted(glob.glob(os.path.join(subfolder_HR_RGB, '*')))
                frames_path_LR_RGB = sorted(glob.glob(os.path.join(subfolder_LR_RGB, '*')))

                assert len(frames_path_HR_RGB) == len(frames_path_LR_RGB), 'Different number of images in LR and HR folders'

                self.data_info_HR[subfolder_HR_RGB] = frames_path_HR_RGB
                self.data_info_LR[subfolder_LR_RGB] = frames_path_LR_RGB
        
    def __getitem__(self, index):

        if self.mode == 'train':

            border = self.data_info['border'][index]
            frame_paths_LR_RGB = []
            frame_paths_HR_RGB = []

            if border == 1:
                frame_paths_LR_RGB = [self.data_info['path_LR_RGB'][index] for _ in range(self.half_N_frames * 2 + 1)]
                frame_paths_HR_RGB = [self.data_info['path_HR_RGB'][index] for _ in range(self.half_N_frames * 2 + 1)]
            else:
                for i in range(self.half_N_frames, -1, -1):
                    frame_paths_LR_RGB.append(self.data_info['path_LR_RGB'][index - i])
                    frame_paths_HR_RGB.append(self.data_info['path_HR_RGB'][index - i])
                for i in range(1, self.half_N_frames + 1):
                    frame_paths_LR_RGB.append(self.data_info['path_LR_RGB'][index + i])
                    frame_paths_HR_RGB.append(self.data_info['path_HR_RGB'][index + i])

            img_LRs_RGB_list = []
            img_HRs_RGB_list = []

            for LR_RGB_path, HR_RGB_path in zip(frame_paths_LR_RGB, frame_paths_HR_RGB):
                # read RGB images
                img_LR_RGB = datautils.read_img(LR_RGB_path, israw=False)
                img_LRs_RGB_list.append(img_LR_RGB)
                img_HR_RGB = datautils.read_img(HR_RGB_path, israw=False)
                img_HRs_RGB_list.append(img_HR_RGB)

            img_LRs_RGB_list, img_HRs_RGB_list = datautils.random_crop_BasicVSR(img_LRs_RGB_list, img_HRs_RGB_list,
                                                                                self.opt.LR_size,
                                                                                self.opt.scale)
        elif self.mode == 'train_val' or self.mode == 'valid':

            video_path_HR_RGB = self.videos_path_HR_RGB[index]
            video_path_LR_RGB = self.videos_path_LR_RGB[index]

            frames_path_HR_RGB = sorted(glob.glob(os.path.join(video_path_HR_RGB, '*')))
            frames_path_LR_RGB = sorted(glob.glob(os.path.join(video_path_LR_RGB, '*')))

            frame_paths_LR_RGB = []
            frame_paths_HR_RGB = []

            for i in range(self.half_N_frames, -1, -1):
                frame_paths_LR_RGB.append(frames_path_LR_RGB[10 - i])
                frame_paths_HR_RGB.append(frames_path_HR_RGB[10 - i])
            for i in range(1, self.half_N_frames + 1):
                frame_paths_LR_RGB.append(frames_path_LR_RGB[10 + i])
                frame_paths_HR_RGB.append(frames_path_HR_RGB[10 + i])

            img_LRs_RGB_list = []
            img_HRs_RGB_list = []
            for LR_RGB_path, HR_RGB_path in zip(frame_paths_LR_RGB, frame_paths_HR_RGB):
                # read RGB images
                img_LR_RGB = datautils.read_img(LR_RGB_path, israw=False)
                img_LRs_RGB_list.append(img_LR_RGB)
                img_HR_RGB = datautils.read_img(HR_RGB_path, israw=False)
                img_HRs_RGB_list.append(img_HR_RGB)

        else:

            video_path_HR_RGB = self.videos_path_HR_RGB[index]
            video_path_LR_RGB = self.videos_path_LR_RGB[index]

            frame_paths_HR_RGB = self.data_info_HR[video_path_HR_RGB]
            frame_paths_LR_RGB = self.data_info_LR[video_path_LR_RGB]

            img_LRs_RGB_list = []
            img_HRs_RGB_list = []
            
            for LR_RGB_path, HR_RGB_path in zip(frame_paths_LR_RGB, frame_paths_HR_RGB):
                # read RGB images
                img_LR_RGB = datautils.read_img(LR_RGB_path, israw=False)
                img_LRs_RGB_list.append(img_LR_RGB)
                img_HR_RGB = datautils.read_img(HR_RGB_path, israw=False)
                img_HRs_RGB_list.append(img_HR_RGB)

        img_LRs_RGB = np.stack(img_LRs_RGB_list, axis=0)
        img_LRs_RGB = torch.from_numpy(img_LRs_RGB).float()
        img_HRs_RGB = np.stack(img_HRs_RGB_list, axis=0)
        img_HRs_RGB = torch.from_numpy(img_HRs_RGB).float()

        return {'LRs_RGB': img_LRs_RGB, 'HRs_RGB': img_HRs_RGB, 'idx': index, 'RGB_gt_name': frame_paths_HR_RGB}

    def __len__(self):

        if self.mode == 'train':
            return len(self.data_info['path_HR_RGB'])
        
        else:
            return len(self.videos_path_HR_RGB)
