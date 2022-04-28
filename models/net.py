import torch
import torch.nn as nn
import torch.nn.functional as F


class CPM_UNet(nn.Module):
    def __init__(self, num_stages, num_joints):
        super(CPM_UNet, self).__init__()
        self.num_stages = num_stages
        
        # replace self.features with a unet architecture
        # self.features = CPM_ImageFeatures()
        self.features = UNet(3)
        
        self.stage1 = CPM_Stage1(num_joints)
        self.stageT = CPM_StageT(num_joints)

    def forward(self, image, center_map):
        heatmaps = []
        stage1_maps = self.stage1(image)
        features = self.features(image)

        heatmaps.append(stage1_maps)

        for _ in range(self.num_stages - 1):
            cur_map = self.stageT(features, heatmaps[-1], center_map)
            heatmaps.append(cur_map)

        return heatmaps


class CPM_ImageFeatures(nn.Module):
    def __init__(self):
        super(CPM_ImageFeatures, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        # x = F.relu(self.conv4(x))
        return x


class CPM_Stage1(nn.Module):
    def __init__(self, num_joints):
        super(CPM_Stage1, self).__init__()
        self.num_joints = num_joints

        # self.features = CPM_ImageFeatures()
        self.features = UNet(3)
        self.conv5 = nn.Conv2d(64, 512, kernel_size=9, padding=4)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7 = nn.Conv2d(512, self.num_joints + 1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        return x


class CPM_StageT(nn.Module):
    def __init__(self, num_joints):
        super(CPM_StageT, self).__init__()
        self.num_joints = num_joints

        self.conv_image = nn.Conv2d(64, 32, kernel_size=5, padding=2)

        self.conv1 = nn.Conv2d(32 + self.num_joints + 2, 128, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(128, self.num_joints + 1, kernel_size=1, padding=0)

    def forward(self, features, prev_map, center_map):
        x = F.relu(self.conv_image(features))
        x = torch.cat([prev_map, x, center_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.inc = Block(n_channels, 64)
        self.down1 = Encode(64, 128)
        self.down2 = Encode(128, 256)
        self.down3 = Encode(256, 512)
        # self.down4 = Encode(512, 1024)
        # self.up1 = Decode(1024, 512)
        self.up2 = Decode(512, 256)
        self.up3 = Decode(256, 128)
        self.up4 = Decode(128, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # out = self.up4(x, x1)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        out = self.up4(x, x1)
        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
        

class Encode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Decode(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Block(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
