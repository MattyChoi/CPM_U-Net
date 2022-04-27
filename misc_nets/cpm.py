import torch
import torch.nn as nn
import torch.nn.functional as F

class CPM(nn.Module):
    def __init__(self, num_stages, num_joints):
        super(CPM, self).__init__()
        self.num_stages = num_stages
        self.heatmaps = []

        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        self.features = CPM_ImageFeatures()
        self.stage1 = CPM_Stage1(num_joints)
        self.stageT = CPM_StageT(num_joints)

    def forward(self, image, center_map):
        pooled_cmap = self.pool_center(center_map)
        stage1_maps = self.stage1(image)
        features = self.features(image)

        self.heatmaps.append(stage1_maps)
        
        for _ in range(self.num_stages - 2):
            cur_map = self.stageT(features, self.heatmaps[-1], pooled_cmap)
            self.heatmaps.append(cur_map)

        return self.heatmaps


class CPM_ImageFeatures(nn.Module):
    def __init__(self):
        super().__init__()
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
        x = F.relu(self.conv4(x))
        return x


class CPM_Stage1(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints

        self.features = CPM_ImageFeatures()
        self.conv5 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
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
        super().__init__()
        self.num_joints = num_joints

        # self.conv_image = nn.Conv2d(128, 32, kernel_size=5, padding=2)

        self.conv1 = nn.Conv2d(32 + self.num_joints + 2, 128, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(128, self.num_joints + 1, kernel_size=1, padding=0)

    def forward(self, features, prev_map, center_map):
        # x = F.relu(self.conv_image(features))
        x = torch.cat([prev_map, features, center_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x