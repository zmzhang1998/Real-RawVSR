import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dcn.deform_conv import ModulatedDeformConvPack as DCN

class PCD_Alignv2(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Alignv2, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, raw_nbr_fea_l, rawpack_nbr_fea_l, rawpack_ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        rawpack_L3_offset = torch.cat([rawpack_nbr_fea_l[2], rawpack_ref_fea_l[2]], dim=1)
        rawpack_L3_offset = self.lrelu(self.L3_offset_conv1(rawpack_L3_offset))
        rawpack_L3_offset = self.lrelu(self.L3_offset_conv2(rawpack_L3_offset))
        raw_L3_offset = F.interpolate(rawpack_L3_offset, scale_factor=2, mode='bilinear', align_corners=False)

        rawpack_L3_fea = self.lrelu(self.L3_dcnpack([rawpack_nbr_fea_l[2], rawpack_L3_offset]))
        raw_L3_fea = self.lrelu(self.L3_dcnpack([raw_nbr_fea_l[2], raw_L3_offset * 2]))

        # L2
        rawpack_L2_offset = torch.cat([rawpack_nbr_fea_l[1], rawpack_ref_fea_l[1]], dim=1)
        rawpack_L2_offset = self.lrelu(self.L2_offset_conv1(rawpack_L2_offset))
        rawpack_L3_offset = F.interpolate(rawpack_L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        rawpack_L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([rawpack_L2_offset, rawpack_L3_offset * 2], dim=1)))
        rawpack_L2_offset = self.lrelu(self.L2_offset_conv3(rawpack_L2_offset))
        raw_L2_offset = F.interpolate(rawpack_L2_offset, scale_factor=2, mode='bilinear', align_corners=False)

        rawpack_L2_fea = self.lrelu(self.L2_dcnpack([rawpack_nbr_fea_l[1], rawpack_L2_offset]))
        raw_L2_fea = self.lrelu(self.L2_dcnpack([raw_nbr_fea_l[1], raw_L2_offset * 2]))

        rawpack_L3_fea = F.interpolate(rawpack_L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        raw_L3_fea = F.interpolate(raw_L3_fea, scale_factor=2, mode='bilinear', align_corners=False)

        rawpack_L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([rawpack_L2_fea, rawpack_L3_fea], dim=1)))
        raw_L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([raw_L2_fea, raw_L3_fea], dim=1)))

        # L1
        rawpack_L1_offset = torch.cat([rawpack_nbr_fea_l[0], rawpack_ref_fea_l[0]], dim=1)
        rawpack_L1_offset = self.lrelu(self.L1_offset_conv1(rawpack_L1_offset))
        rawpack_L2_offset = F.interpolate(rawpack_L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        rawpack_L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([rawpack_L1_offset, rawpack_L2_offset * 2], dim=1)))
        rawpack_L1_offset = self.lrelu(self.L1_offset_conv3(rawpack_L1_offset))
        raw_L1_offset = F.interpolate(rawpack_L1_offset, scale_factor=2, mode='bilinear', align_corners=False)

        rawpack_L1_fea = self.lrelu(self.L1_dcnpack([rawpack_nbr_fea_l[0], rawpack_L1_offset]))
        raw_L1_fea = self.lrelu(self.L1_dcnpack([raw_nbr_fea_l[0], raw_L1_offset * 2]))

        rawpack_L2_fea = F.interpolate(rawpack_L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        raw_L2_fea = F.interpolate(raw_L2_fea, scale_factor=2, mode='bilinear', align_corners=False)

        rawpack_L1_fea = self.lrelu(self.L1_fea_conv(torch.cat([rawpack_L1_fea, rawpack_L2_fea], dim=1)))
        raw_L1_fea = self.lrelu(self.L1_fea_conv(torch.cat([raw_L1_fea, raw_L2_fea], dim=1)))

        # Cascading
        rawpack_offset = torch.cat([rawpack_L1_fea, rawpack_ref_fea_l[0]], dim=1)
        rawpack_offset = self.lrelu(self.cas_offset_conv1(rawpack_offset))
        rawpack_offset = self.lrelu(self.cas_offset_conv2(rawpack_offset))
        raw_offset = F.interpolate(rawpack_offset, scale_factor=2, mode='bilinear', align_corners=False)

        rawpack_L1_fea = self.lrelu(self.cas_dcnpack([rawpack_L1_fea, rawpack_offset]))
        raw_L1_fea = self.lrelu(self.cas_dcnpack([raw_L1_fea, raw_offset * 2]))

        return raw_L1_fea, rawpack_L1_fea

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea
