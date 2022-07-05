
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
            self.paths_LR_RAW = opt.train_paths_LR_RAW
            self.paths_LR_RGB = opt.train_paths_LR_RGB
        else:
            self.paths_HR_RGB = opt.test_paths_HR_RGB
            self.paths_LR_RAW = opt.test_paths_LR_RAW
            self.paths_LR_RGB = opt.test_paths_LR_RGB

        self.videos_path_HR_RGB = sorted(glob.glob(os.path.join(self.paths_HR_RGB, '*')))
        self.videos_path_LR_RAW = sorted(glob.glob(os.path.join(self.paths_LR_RAW, '*')))
        self.videos_path_LR_RGB = sorted(glob.glob(os.path.join(self.paths_LR_RGB, '*')))

        if  self.mode == 'train' or self.mode == 'test':

            self.data_info = {'path_LR_RAW': [], 'path_LR_RGB': [], 'path_HR_RGB': [], 'border': []}

            for subfolder_LR_RAW, subfolder_HR_RGB, subfolder_LR_RGB in zip(self.videos_path_LR_RAW,
                                                                            self.videos_path_HR_RGB,
                                                                            self.videos_path_LR_RGB):
                frames_path_LR_RAW = sorted(glob.glob(os.path.join(subfolder_LR_RAW, '*')))
                frames_path_HR_RGB = sorted(glob.glob(os.path.join(subfolder_HR_RGB, '*')))
                frames_path_LR_RGB = sorted(glob.glob(os.path.join(subfolder_LR_RGB, '*')))

                assert len(frames_path_LR_RAW) == len(frames_path_LR_RGB), 'Different number of images in LR and HR folders'

                self.data_info['path_LR_RAW'].extend(frames_path_LR_RAW)
                self.data_info['path_HR_RGB'].extend(frames_path_HR_RGB)
                self.data_info['path_LR_RGB'].extend(frames_path_LR_RGB)

                is_border = [0] * len(frames_path_LR_RAW)
                for i in range(self.half_N_frames):
                    is_border[i] = 1
                    is_border[len(frames_path_LR_RAW) - i - 1] = 1
                self.data_info['border'].extend(is_border)

    def __getitem__(self, index):

        if self.mode == 'train' or self.mode == 'test':
            
            border = self.data_info['border'][index]
            frame_paths_LR_RAW = []
            frame_path_HR_RGB = self.data_info['path_HR_RGB'][index]
            frame_path_LR_RGB = self.data_info['path_LR_RGB'][index]

            if border == 1:
                frame_paths_LR_RAW = [self.data_info['path_LR_RAW'][index] for _ in range(self.half_N_frames * 2 + 1)]
            else:
                for i in range(self.half_N_frames, -1, -1):
                    frame_paths_LR_RAW.append(self.data_info['path_LR_RAW'][index - i])
                for i in range(1, self.half_N_frames + 1):
                    frame_paths_LR_RAW.append(self.data_info['path_LR_RAW'][index + i])

            img_HR_RGB = datautils.read_img(frame_path_HR_RGB, israw=False)
            img_LR_RGB = datautils.read_img(frame_path_LR_RGB, israw=False)

            img_LRs_RAW_list = []
            for LR_RAW_path in frame_paths_LR_RAW:
                # read LR_RAW images
                img_LR_RAW = datautils.read_img(LR_RAW_path, israw=True)
                img_LRs_RAW_list.append(img_LR_RAW)

        else:
            video_path_HR_RGB = self.videos_path_HR_RGB[index]
            video_path_LR_RAW = self.videos_path_LR_RAW[index]
            video_path_LR_RGB = self.videos_path_LR_RGB[index]

            frames_path_HR_RGB = sorted(glob.glob(os.path.join(video_path_HR_RGB, '*')))
            frames_path_LR_RAW = sorted(glob.glob(os.path.join(video_path_LR_RAW, '*')))
            frames_path_LR_RGB = sorted(glob.glob(os.path.join(video_path_LR_RGB, '*')))

            frame_paths_LR_RAW = []
            frame_path_HR_RGB = frames_path_HR_RGB[10]
            frame_path_LR_RGB = frames_path_LR_RGB[10]

            for i in range(self.half_N_frames, -1, -1):
                frame_paths_LR_RAW.append(frames_path_LR_RAW[10 - i])
            for i in range(1, self.half_N_frames + 1):
                frame_paths_LR_RAW.append(frames_path_LR_RAW[10 + i])

            img_HR_RGB = datautils.read_img(frame_path_HR_RGB, israw=False)
            img_LR_RGB = datautils.read_img(frame_path_LR_RGB, israw=False)

            img_LRs_RAW_list = []
            for LR_RAW_path in frame_paths_LR_RAW:
                # read LR_RAW images
                img_LR_RAW = datautils.read_img(LR_RAW_path, israw=True)
                img_LRs_RAW_list.append(img_LR_RAW)

        if self.mode == 'train':
            img_LRs_RAW_list, img_LR_RGB, img_HR_RGB = datautils.random_crop(img_LRs_RAW_list, img_LR_RGB,
                                                                             img_HR_RGB,
                                                                             self.opt.LR_size,
                                                                             self.opt.scale)
        
        img_LRs_RAW_nopack = np.stack(img_LRs_RAW_list, axis=0)

        img_LRs_RAW = datautils.pack_rggb_raws(img_LRs_RAW_nopack)
        img_LRs_RAW_nopack = torch.from_numpy(img_LRs_RAW_nopack).float()
        img_LRs_RAW = torch.from_numpy(img_LRs_RAW).float()
        img_HR_RGB = torch.from_numpy(img_HR_RGB).float()
        img_LR_RGB = torch.from_numpy(img_LR_RGB).float()

        return {'LRs_RAW': img_LRs_RAW, 'LRs_RAW_nopack': img_LRs_RAW_nopack,
                'HR_RGB': img_HR_RGB, 'LR_RGB': img_LR_RGB, 'idx': index,
                'RGB_gt_name': frame_path_HR_RGB}

    def __len__(self):
        if self.mode == 'test' or self.mode == 'train':
            return len(self.data_info['path_HR_RGB'])
        else:
            return len(self.videos_path_HR_RGB)
