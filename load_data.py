from torch.utils.data import Dataset
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import utils

dataset_dir = './data'

class OMC(Dataset):
    """
    Dataset for OMC
    """
    def __init__(self, is_training=True):
        super(OMC, self).__init__()
        self.is_training = is_training

        train_json_dir = os.path.join(dataset_dir, 'train_annotations.json')
        val_json_dir = os.path.join(dataset_dir, 'val_annotations.json')

        with open(train_json_dir) as f:
            dic = json.load(f)
            self.train_list = [item for item in dic['data']]

        with open(val_json_dir) as f:
            dic = json.load(f)
            self.val_list = [item for item in dic['data']]


    def __getitem__(self, index):
        if self.is_training:
            ele_anno = self.anno[self.train_list[index]]
        else:
            ele_anno = self.anno[self.test_list[index]]

        img_sz = (368,368)
        img_folder_dir = os.path.join(dataset_dir, 'train')
        img_dir = os.path.join(img_folder_dir, ele_anno['img_paths'])
        img = dataset_dir.load_img(img_dir)

        pts = ele_anno['joint_self']
        cen = ele_anno['objpos'].copy()
        scale = ele_anno['scale_provided']

        # generate crop image
        #print(img)
        img_crop, pts_crop, cen_crop = utils.crop(img, ele_anno)
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
        if self.is_training:
            return len(self.train_list)
        return len(self.val_list)

    def collate_fn(self, batch):
        imgs, heatmaps, centermap = list(zip(*batch))

        imgs = np.stack(imgs, axis=0)
        heatmaps = np.stack(heatmaps, axis=0)
        centermap = np.stack(centermap, axis=0)

        return imgs, heatmaps, centermap