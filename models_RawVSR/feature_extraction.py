import torch
import torch.nn as nn

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()

        self.conv1_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        conv1_1 = self.lrelu(self.conv1_1(x))
        conv1_2 = self.lrelu(self.conv1_2(conv1_1))

        conv2_1 = self.lrelu(self.conv2_1(conv1_2))
        conv2_2 = self.lrelu(self.conv2_2(conv2_1))

        conv3_1 = self.lrelu(self.conv3_1(conv2_2))
        conv3_2 = self.lrelu(self.conv3_2(conv3_1))

        conv4_1 = self.lrelu(self.conv4_1(conv3_2, conv2_2.size()))
        conv4_2 = self.lrelu(self.conv4_2(torch.cat([conv2_2, conv4_1], dim=1)))
        conv4_3 = self.lrelu(self.conv4_3(conv4_2))

        conv5_1 = self.lrelu(self.conv5_1(conv4_3, conv1_2.size()))
        conv5_2 = self.lrelu(self.conv5_2(torch.cat([conv1_2, conv5_1], dim=1)))
        out = self.lrelu(self.conv5_3(conv5_2))

        return out
