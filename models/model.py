import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from models.align_net import PCD_Alignv2
from models.backbone import make_layer, RB, SKConv, BIMv1, BIMv2, TAM_Module, TSA_Fusion

class RRVSR(nn.Module):
    def __init__(self, nf=64, nframes=5, scale=4, groups=8):
        super(RRVSR, self).__init__()

        # parameters
        self.center = nframes // 2
        self.scale = scale
        self.nf = nf

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        # Feature Extraction
        self.raw_conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.rawpack_conv_first = nn.Conv2d(4, nf, 3, 1, 1, bias=True)

        RB_begin = functools.partial(RB, nf=nf)
        self.fea_extraction1 = make_layer(RB_begin, 5)
        self.fea_extraction2 = make_layer(RB_begin, 5)

        self.rawfea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.rawfea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.rawfea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.rawfea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.rawpackfea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.rawpackfea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.rawpackfea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.rawpackfea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 

        self.pcd_align = PCD_Alignv2(nf=nf, groups=groups)
        
        self.rawpack_up = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            self.lrelu,
            nn.Conv2d(nf, nf, 3, 1, 1),
            self.lrelu
        )

        self.BIM1 = BIMv1(nf)
        self.BIM2 = BIMv2(nf)
        
        # attention
        self.t_attention1 = TAM_Module()
        self.t_fusion1 = TSA_Fusion(nf, 2*nframes, self.center)
        self.t_attention2 = TAM_Module()
        self.t_fusion2 = TSA_Fusion(nf, 2*nframes, self.center)

        self.skconv = SKConv(nf, M=2, r=2, L=32)

        # build backbone
        RB_end = functools.partial(RB, nf=nf)
        self.recon_RB = make_layer(RB_end, 10)

        # upsampling
        if scale == 4:
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.sr_conv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.sr_conv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.sr_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.sr_conv4 = nn.Conv2d(nf, 3, 3, 1, 1)

            self.raw_skip_pixel_shuffle = nn.PixelShuffle(2)
            self.raw_skipup1 = nn.Conv2d(1, nf * 4, 3, 1, 1, bias=True)
            self.raw_skipup2 = nn.Conv2d(nf, 3 * 4, 3, 1, 1, bias=True)

            self.rawpack_skip_pixel_shuffle = nn.PixelShuffle(2)
            self.rawpack_skipup1 = nn.Conv2d(4, nf * 4, 3, 1, 1, bias=True)
            self.rawpack_skipup2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.rawpack_skipup3 = nn.Conv2d(nf, 3 * 4, 3, 1, 1, bias=True)

        elif scale == 3:
            self.pixel_shuffle = nn.PixelShuffle(3)
            self.sr_conv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1)
            self.sr_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.sr_conv3 = nn.Conv2d(nf, 3, 3, 1, 1)

            self.raw_skip_pixel_shuffle = nn.PixelShuffle(3)
            self.raw_skipup1 = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
            self.raw_skipup2 = nn.Conv2d(nf, 3 * 9, 3, 1, 1, bias=True)

            self.rawpack_skip_pixel_shuffle1 = nn.PixelShuffle(2)
            self.rawpack_skip_pixel_shuffle2 = nn.PixelShuffle(3)
            self.rawpack_skipup1 = nn.Conv2d(4, nf * 4, 3, 1, 1, bias=True)
            self.rawpack_skipup2 = nn.Conv2d(nf, 3 * 9, 3, 1, 1, bias=True)

        elif scale == 2:
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.sr_conv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.sr_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.sr_conv3 = nn.Conv2d(nf, 3, 3, 1, 1)

            self.raw_skip_pixel_shuffle = nn.PixelShuffle(2)
            self.raw_skipup1 = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
            self.raw_skipup2 = nn.Conv2d(nf, 3 * 4, 3, 1, 1, bias=True)

            self.rawpack_skip_pixel_shuffle = nn.PixelShuffle(2)
            self.rawpack_skipup1 = nn.Conv2d(4, nf * 4, 3, 1, 1, bias=True)
            self.rawpack_skipup3 = nn.Conv2d(nf, 3 * 4, 3, 1, 1, bias=True)

        else:
            raise Exception('scale {} is not supported!'.format(scale))

        # self.isp = ISP()

    def forward(self, raw, rawpack):
        # N video frames
        B, N, C, H, W = raw.size()
        raw_center = raw[:, self.center, :, :, :].contiguous()
        rawpack_center = rawpack[:, self.center, :, :, :].contiguous()

        # skip module
        if self.scale == 4:
            raw_skip1 = self.lrelu(self.raw_skip_pixel_shuffle(self.raw_skipup1(raw_center)))
            raw_skip2 = self.raw_skip_pixel_shuffle(self.raw_skipup2(raw_skip1))

            rawpack_skip1 = self.lrelu(self.rawpack_skip_pixel_shuffle(self.rawpack_skipup1(rawpack_center)))
            rawpack_skip2 = self.lrelu(self.rawpack_skip_pixel_shuffle(self.rawpack_skipup2(rawpack_skip1)))
            rawpack_skip3 = self.rawpack_skip_pixel_shuffle(self.rawpack_skipup3(rawpack_skip2))

        elif self.scale == 3:
            raw_skip1 = self.lrelu(self.raw_skipup1(raw_center))
            raw_skip2 = self.raw_skip_pixel_shuffle(self.raw_skipup2(raw_skip1))
            
            rawpack_skip1 = self.lrelu(self.rawpack_skip_pixel_shuffle1(self.rawpack_skipup1(rawpack_center)))
            rawpack_skip2 = self.rawpack_skip_pixel_shuffle2(self.rawpack_skipup2(rawpack_skip1))

        elif self.scale == 2:
            raw_skip1 = self.lrelu(self.raw_skipup1(raw_center))
            raw_skip2 = self.raw_skip_pixel_shuffle(self.raw_skipup2(raw_skip1))
            
            rawpack_skip1 = self.lrelu(self.rawpack_skip_pixel_shuffle(self.rawpack_skipup1(rawpack_center)))
            rawpack_skip2 = self.rawpack_skip_pixel_shuffle(self.rawpack_skipup3(rawpack_skip1))
        
        else:
            raise Exception('scale {} is not supported!'.format(self.scale))      
        
        # extract LR features
        rawL1_fea = self.lrelu(self.raw_conv_first(raw.view(-1, C, H, W)))
        rawL1_fea = self.fea_extraction1(rawL1_fea)
        # L2
        rawL2_fea = self.lrelu(self.rawfea_L2_conv1(rawL1_fea))
        rawL2_fea = self.lrelu(self.rawfea_L2_conv2(rawL2_fea))
        # L3
        rawL3_fea = self.lrelu(self.rawfea_L3_conv1(rawL2_fea))
        rawL3_fea = self.lrelu(self.rawfea_L3_conv2(rawL3_fea)) 

        rawL1_fea = rawL1_fea.view(B, N, -1, H, W)
        rawL2_fea = rawL2_fea.view(B, N, -1, H // 2, W // 2)
        rawL3_fea = rawL3_fea.view(B, N, -1, H // 4, W // 4)

        rawpackL1_fea = self.lrelu(self.rawpack_conv_first(rawpack.view(-1, C*4, H // 2, W // 2)))
        rawpackL1_fea = self.fea_extraction2(rawpackL1_fea)
        # L2
        rawpackL2_fea = self.lrelu(self.rawpackfea_L2_conv1(rawpackL1_fea))
        rawpackL2_fea = self.lrelu(self.rawpackfea_L2_conv2(rawpackL2_fea))
        # L3
        rawpackL3_fea = self.lrelu(self.rawpackfea_L3_conv1(rawpackL2_fea))
        rawpackL3_fea = self.lrelu(self.rawpackfea_L3_conv2(rawpackL3_fea)) 

        rawpackL1_fea = rawpackL1_fea.view(B, N, -1, H // 2, W // 2)
        rawpackL2_fea = rawpackL2_fea.view(B, N, -1, H // 4, W // 4)
        rawpackL3_fea = rawpackL3_fea.view(B, N, -1, H // 8, W // 8)

        #### PCD align
        # ref feature list
        rawpack_ref_fea_l = [
            rawpackL1_fea[:, self.center, :, :, :].clone(), rawpackL2_fea[:, self.center, :, :, :].clone(),
            rawpackL3_fea[:, self.center, :, :, :].clone()
        ]
        rawpack_aligned_fea = []
        raw_aligned_fea = []

        for i in range(N):
            raw_nbr_fea_l = [
                rawL1_fea[:, i, :, :, :].clone(), rawL2_fea[:, i, :, :, :].clone(),
                rawL3_fea[:, i, :, :, :].clone()
            ]
            rawpack_nbr_fea_l = [
                rawpackL1_fea[:, i, :, :, :].clone(), rawpackL2_fea[:, i, :, :, :].clone(),
                rawpackL3_fea[:, i, :, :, :].clone()
            ]
            raw_fea_l, rawpack_fea_l = self.pcd_align(raw_nbr_fea_l, rawpack_nbr_fea_l, rawpack_ref_fea_l)

            raw_aligned_fea.append(raw_fea_l)
            rawpack_aligned_fea.append(rawpack_fea_l)
        
        raw_fea = torch.stack(raw_aligned_fea, dim=1)
        rawpack_fea = torch.stack(rawpack_aligned_fea, dim=1)

        BIM_raw_fea = self.BIM1(raw_fea, rawpack_fea)
        BIM_rawpack_fea = self.BIM2(rawpack_fea, raw_fea)

        raw_fea = BIM_raw_fea.permute(0, 2, 1, 3, 4).contiguous().view(-1, 2*N, H, W)
        raw_fea = self.t_attention1(raw_fea).view(B, -1, 2*N, H, W).permute(0, 2, 1, 3, 4).contiguous()
        raw_fea = self.t_fusion1(raw_fea)
        
        rawpack_fea = BIM_rawpack_fea.permute(0, 2, 1, 3, 4).contiguous().view(-1, 2*N, H//2, W//2)
        rawpack_fea = self.t_attention2(rawpack_fea).view(B, -1, 2*N, H//2, W//2).permute(0, 2, 1, 3, 4).contiguous()
        rawpack_fea = self.t_fusion2(rawpack_fea)

        rawpack_fea = self.rawpack_up(rawpack_fea)

        fea = self.skconv(raw_fea, rawpack_fea)

        fea = self.recon_RB(fea)  # [B, nf, H, W]
        
        # upsampling
        if self.scale == 4:
            sr = self.lrelu(self.pixel_shuffle(self.sr_conv1(fea)))
            sr = sr + raw_skip1 + rawpack_skip2
            sr = self.lrelu(self.pixel_shuffle(self.sr_conv2(sr)))
            sr = self.lrelu(self.sr_conv3(sr))
            sr = self.sr_conv4(sr)
            sr = sr + raw_skip2 + rawpack_skip3

        elif self.scale == 3:
            sr = self.lrelu(self.pixel_shuffle(self.sr_conv1(fea)))
            sr = self.lrelu(self.sr_conv2(sr))
            sr = self.sr_conv3(sr)
            sr = sr + raw_skip2 + rawpack_skip2

        elif self.scale == 2:
            sr = self.lrelu(self.pixel_shuffle(self.sr_conv1(fea)))
            sr = self.lrelu(self.sr_conv2(sr))
            sr = self.sr_conv3(sr)
            sr = sr + raw_skip2 + rawpack_skip2

        else:
            raise Exception('scale {} is not supported!'.format(self.scale))

        return sr
