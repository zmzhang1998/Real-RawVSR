
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
import data.data_utils as datautils
import glob
import os

class myData(Dataset):

    def __init__(self, opt, mode):
        super(myData, self).__init__()
        self.opt = opt
        self.mode = mode
        self.half_N_frames = opt.N_frames // 2

        if self.mode == 'train' or self.mode == 'train_val':

            self.paths_HR_RGB = opt.train_paths_HR_RGB
            self.paths_LR_RGB = opt.train_paths_LR_RGB
        else:
            self.paths_HR_RGB = opt.test_paths_HR_RGB
            self.paths_LR_RGB = opt.test_paths_LR_RGB

        self.videos_path_HR_RGB = sorted(glob.glob(os.path.join(self.paths_HR_RGB, '*')))
        self.videos_path_LR_RGB = sorted(glob.glob(os.path.join(self.paths_LR_RGB, '*')))

        if  self.mode == 'train' or self.mode == 'test':

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

    def __getitem__(self, index):

        if self.mode == 'train' or self.mode == 'test':
            
            border = self.data_info['border'][index]
            frame_paths_LR_RGB = []
            frame_path_HR_RGB = self.data_info['path_HR_RGB'][index]

            if border == 1:
                frame_paths_LR_RGB = [self.data_info['path_LR_RGB'][index] for _ in range(self.half_N_frames * 2 + 1)]
            else:
                for i in range(self.half_N_frames, -1, -1):
                    frame_paths_LR_RGB.append(self.data_info['path_LR_RGB'][index - i])
                for i in range(1, self.half_N_frames + 1):
                    frame_paths_LR_RGB.append(self.data_info['path_LR_RGB'][index + i])

            img_HR_RGB = datautils.read_img(frame_path_HR_RGB, israw=False)

            img_LRs_RGB_list = []
            for LR_RGB_path in frame_paths_LR_RGB:
                # read LR_RAW images
                img_LR_RGB = datautils.read_img(LR_RGB_path, israw=False)
                img_LRs_RGB_list.append(img_LR_RGB)

        else:
            video_path_HR_RGB = self.videos_path_HR_RGB[index]
            video_path_LR_RGB = self.videos_path_LR_RGB[index]

            frames_path_HR_RGB = sorted(glob.glob(os.path.join(video_path_HR_RGB, '*')))
            frames_path_LR_RGB = sorted(glob.glob(os.path.join(video_path_LR_RGB, '*')))

            frame_paths_LR_RGB = []
            frame_path_HR_RGB = frames_path_HR_RGB[10]

            for i in range(self.half_N_frames, -1, -1):
                frame_paths_LR_RGB.append(frames_path_LR_RGB[10 - i])
            for i in range(1, self.half_N_frames + 1):
                frame_paths_LR_RGB.append(frames_path_LR_RGB[10 + i])

            img_HR_RGB = datautils.read_img(frame_path_HR_RGB, israw=False)

            img_LRs_RGB_list = []
            for LR_RGB_path in frame_paths_LR_RGB:
                # read LR_RAW images
                img_LR_RGB = datautils.read_img(LR_RGB_path, israw=False)
                img_LRs_RGB_list.append(img_LR_RGB)

        if self.mode == 'train':
            img_LRs_RGB_list, img_HR_RGB = datautils.random_crop_EDVR(img_LRs_RGB_list, img_HR_RGB,
                                                                      self.opt.LR_size,
                                                                      self.opt.scale)
        
        img_LRs_RGB = np.stack(img_LRs_RGB_list, axis=0)

        img_LRs_RGB = torch.from_numpy(img_LRs_RGB).float()
        img_HR_RGB = torch.from_numpy(img_HR_RGB).float()

        return {'LRs_RGB': img_LRs_RGB, 'HR_RGB': img_HR_RGB, 'idx': index,
                'RGB_gt_name': frame_path_HR_RGB}

    def __len__(self):
        if self.mode == 'test' or self.mode == 'train':
            return len(self.data_info['path_HR_RGB'])
        else:
            return len(self.videos_path_HR_RGB)
