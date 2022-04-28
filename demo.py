import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np

import json
import os

from models.cpm import CPM
from models.net import CPM_UNet
from load_data import test_OMC
from utils import show_heatmaps, get_landmarks_from_preds, visualize_result

cuda = torch.cuda.is_available()

def demo(model):
    num_joints = 17

    model.eval()
    test_dataset = test_OMC()
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)

    for i, (img, cmap, bbox) in enumerate(test_loader):
        test_img = torch.FloatTensor(img)
        cmap = torch.FloatTensor(cmap)
        bbox = bbox[0]

        pred_heatmaps = model(test_img, cmap)

        if i > 1:
            pred_hmap = pred_heatmaps[-1][0].cpu().detach().numpy().transpose((1,2,0))
            img = img[0].transpose((1,2,0))
            show_heatmaps(img, pred_hmap)
            visualize_result(img, pred_hmap)
        print(i)
            

def main():
    # MODEL_DIR = os.path.join('weights', 'cpm_unet.pkl')
    MODEL_DIR = os.path.join('weights', 'cpm_epoch_3_best.pkl')
    
    # model = CPM_UNet(num_stages=3, num_joints=17)
    model = CPM(num_stages=3, num_joints=17)
    
    model.load_state_dict(torch.load(MODEL_DIR))

    demo(model)



if  __name__ == "__main__":
    main()