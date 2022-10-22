import torch
import torch.nn as nn
from models.dcn.deform_conv import ModulatedDeformConvPack as DCN

class AlignBlock(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(AlignBlock, self).__init__()

        self.offset_fea_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_fea_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.offset_ref_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_ref_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.fuse1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.fuse2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.nf = nf

    def forward(self, fea_l, ref_fea_l):
        # concat for offset
        offset_fea = torch.cat([fea_l, ref_fea_l], dim=1)
        offset_fea = self.lrelu(self.offset_fea_conv1(offset_fea))
        offset_fea = self.lrelu(self.offset_fea_conv2(offset_fea))
        deform_fea_l = self.lrelu(self.dcnpack([fea_l, offset_fea]))

        offset_ref = torch.cat([ref_fea_l, fea_l], dim=1)
        offset_ref = self.lrelu(self.offset_ref_conv1(offset_ref))
        offset_ref = self.lrelu(self.offset_ref_conv2(offset_ref))
        deform_ref_fea_l = self.lrelu(self.dcnpack([ref_fea_l, offset_ref]))

        fea_cat = torch.cat([deform_fea_l, deform_ref_fea_l], dim=1)
        out = self.lrelu(self.fuse2(self.lrelu(self.fuse1(fea_cat))))

        return out


class AlignNet(nn.Module):
    def __init__(self, nf=64, groups=8, num_frames=5):
        super(AlignNet, self).__init__()
        self.align_block = AlignBlock(nf, groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.num_frames = num_frames
        self.num_features = nf

    def forward(self, fea_ls):

        num_frames = self.num_frames

        # from left alignment
        for i in range(num_frames // 2):
            fea_ls[i + 1] = self.align_block(fea_ls[i], fea_ls[i + 1])

        fea_ls_left = fea_ls[i + 1]
        assert i + 1 == num_frames // 2

        # from right alignment
        if num_frames == 3:
            i = num_frames  # only for 3 frames, or error

        for i in range(num_frames - 1, num_frames // 2 + 1, -1):
            fea_ls[i - 1] = self.align_block(fea_ls[i], fea_ls[i - 1])

        fea_ls_right = fea_ls[i - 1]
        assert i - 1 == num_frames // 2 + 1

        central_fea_l = self.align_block(fea_ls_left, fea_ls_right)

        return central_fea_l