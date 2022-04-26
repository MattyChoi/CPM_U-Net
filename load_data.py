from torch.utils.data import Dataset
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import utils

dataset_dir = r'.\data'

class OMC(Dataset):
    """
    Dataset for OMC
    """

    def generate_heatmap(heatmap, pt, sigma_valu=2):
        '''
        :param heatmap: should be a np zeros array with shape (H,W) (only 1 channel), not (H,W,1)
        :param pt: point coords, np array
        :param sigma: should be a tuple with odd values (obsolete)
        :param sigma_valu: value for gaussian blur
        :return: a np array of one joint heatmap with shape (H,W)
        This function is obsolete, use 'generate_heatmaps()' instead.
        '''
        heatmap[int(pt[1])][int(pt[0])] = 1
        heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)
        am = np.max(heatmap)
        heatmap = heatmap/am
        return heatmap



    def generate_heatmaps(img, pts, sigma_valu=2):
        '''
        Generate 16 heatmaps
        :param img: np arrray img, (H,W,C)
        :param pts: joint points coords, np array, same resolu as img
        :param sigma: should be a tuple with odd values (obsolete)
        :param sigma_valu: vaalue for gaussian blur
        :return: np array heatmaps, (H,W,num_pts)
        '''
        H, W = img.shape[0], img.shape[1]
        num_pts = len(pts)
        heatmaps = np.zeros((H, W, num_pts + 1))
        for i, pt in enumerate(pts):
            # Filter unavailable heatmaps
            if pt[0] == 0 and pt[1] == 0:
                continue
            # Filter some points out of the image
            if pt[0] >= W:
                pt[0] = W-1
            if pt[1] >= H:
                pt[1] = H-1
            heatmap = heatmaps[:, :, i]
            heatmap[int(pt[1])][int(pt[0])] = 1  # reverse sequence
            heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)  ##(H,W,1) -> (H,W)
            am = np.max(heatmap)
            heatmap = heatmap / am  # scale to [0,1]
            heatmaps[:, :, i] = heatmap

        heatmaps[:, :, num_pts] = 1.0 - np.max(heatmaps[:, :, :num_pts], axis=2) # add background dim

        return heatmaps


    def __init__(self, is_training=True):
        super(OMC, self).__init__()
        self.is_training = is_training

        if self.is_training:
            dir = os.path.join(dataset_dir, 'train_annotations.json')
        else:
            dir = os.path.join(dataset_dir, 'val_annotations.json')

        with open(dir) as f:
            dic = json.load(f)
            self.feature_list = [item for item in dic['data']]


    def __getitem__(self, index):
        features = self.feature_list[index]

        img_sz = (368,368)
        img_folder_dir = os.path.join(dataset_dir, 'train')
        img_dir = os.path.join(img_folder_dir, features['img_paths'])
        img = dataset_dir.load_img(img_dir)

        pts = features['joint_self']
        cen = features['objpos'].copy()
        scale = features['scale_provided']

        # generate crop image
        #print(img)
        img_crop, pts_crop, cen_crop = utils.crop(img, features)
        pts_crop = np.array(pts_crop)
        cen_crop = np.array(cen_crop)
        
        height, width, _ = img_crop.shape
        train_img = np.transpose(img_crop, (2,0,1))/255.0

        train_heatmaps = utils.generate_heatmaps(np.zeros((46,46)), pts_crop/8)
        train_heatmaps = np.transpose(train_heatmaps, (2,0,1))

        train_centermap = utils.generate_heatmap(np.zeros((368,368)), cen_crop)
        train_centermap = np.expand_dims(train_centermap, axis=0)

        return train_img, train_heatmaps, train_centermap


    def __len__(self):
        return len(self.feature_list)


    def collate_fn(self, batch):
        imgs, heatmaps, centermap = list(zip(*batch))

        imgs = np.stack(imgs, axis=0)
        heatmaps = np.stack(heatmaps, axis=0)
        centermap = np.stack(centermap, axis=0)

        return imgs, heatmaps, centermap