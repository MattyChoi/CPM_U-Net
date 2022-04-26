from torch.utils.data import Dataset
import torch.utils.data as torchdata
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils

dataset_dir = "./data"

class OMC(Dataset):
    """
    Dataset for OMC
    """

    def __init__(self, is_training=True):
        super(OMC, self).__init__()
        self.is_training = is_training

        if self.is_training:
            dir = os.path.join(dataset_dir, 'train_annotation.json')
        else:
            dir = os.path.join(dataset_dir, 'val_annotation.json')

        with open(dir) as f:
            dic = json.load(f)
            self.feature_list = [item for item in dic['data']]


    def __getitem__(self, index):
        features = self.feature_list[index]

        img_sz = (256,256)
        if(self.is_training==True):
            img_folder_dir = os.path.join(dataset_dir, 'train')
        else:
            img_folder_dir = os.path.join(dataset_dir, 'val')
        img_dir = os.path.join(img_folder_dir, features['file'])
        img = mpimg.imread(img_dir)

        # generate crop image
        #print(img)
        img_crop, pts_crop, cen_crop = utils.crop(img, features)
        pts_crop = np.array(pts_crop)
        cen_crop = np.array(cen_crop)
        
        height, width, _ = img_crop.shape
        train_img = np.transpose(img_crop, (2,0,1))/255.0

        train_heatmaps = utils.gen_hmaps(np.zeros((256,256)), pts_crop)
        train_heatmaps = np.transpose(train_heatmaps, (2,0,1))

        train_centermap = utils.gen_cmap(np.zeros((256,256)), cen_crop)
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

def main():
    #plt.ion()
    omc = OMC(is_training=True)
    dataloader = torchdata.DataLoader(omc, batch_size=1, shuffle=True, collate_fn=omc.collate_fn)
    for i, (img, heatmap, centermap) in enumerate(dataloader):
        
        #print(img.shape, heatmap.shape, centermap.shape)
        #print(img_crop[0].shape)

        #imgutils.show_stack_joints(img_crop[0], pts_crop[0], cen_crop[0], num_fig=2*i+1)
        #imgutils.show_stack_joints(img[0], pts[0], cen[0], num_fig=2*i+2)
        utils.show_heatmaps(img[0].transpose(1,2,0), heatmap[0].transpose(1,2,0))
        #plt.pause(5)
        if i == 0:
            break
    #plt.ioff()

if __name__ == "__main__":
    main()