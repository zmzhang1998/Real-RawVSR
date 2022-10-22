import torch
import torch.nn as nn

from models_RawVSR.blocks import SARDB, ChannelAttention

class BackBone(nn.Module):
    def __init__(self, nf, num_stage, num_rdb=2, num_dense_layer=3, growth_rate=64):
        super(BackBone, self).__init__()
        self.stages = nn.ModuleDict()

        for i in range(num_stage):
            self.stages.update({'{}'.format(i): SARDB(nf, nf, num_rdb, num_dense_layer, growth_rate)})

        total_channels = (num_stage + 1) * nf

        self.CA = ChannelAttention(total_channels)
        self.conv_1x1 = nn.Conv2d(total_channels, nf, kernel_size=1, padding=0)
        self.num_stage = num_stage

    def forward(self, x):
        val_list = [None for _ in range(self.num_stage + 1)]
        val_list[0] = x

        for i in range(self.num_stage):
            val_list[i + 1] = self.stages['{}'.format(i)](val_list[i])

        out = torch.cat(val_list, 1)

        out_ca = self.CA(out)
        out = out * out_ca
        out = self.conv_1x1(out)

        return out
