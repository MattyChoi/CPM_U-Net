import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np

import json
import os

from models.cpm import CPM
from models.net import CPM_UNet
from load_data import sanity_check_OMC
from utils import AverageMeter, show_heatmaps, get_landmarks_from_preds

cuda = torch.cuda.is_available()

def test(device, model):
    num_joints = 17

    model.eval()
    test_dataset = sanity_check_OMC()
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)

    for i, (img, cmap, bbox, landmarks) in enumerate(test_loader):
        img = torch.FloatTensor(img).to(device)
        cmap = torch.FloatTensor(cmap).to(device)
        bbox = bbox[0]
        landmarks = landmarks[0]

        pred_heatmaps = model(img, cmap)
        pred_hmap = pred_heatmaps[-1][0].cpu().detach().numpy().transpose((1,2,0))[:,:,:num_joints]
        show_heatmaps(img[0].cpu().detach().numpy().transpose((1,2,0)), pred_hmap)

        pred_landmarks = get_landmarks_from_preds(pred_hmap, bbox, num_joints=17)

        print(pred_landmarks)
        print(landmarks)
        print('--------------------------------------------------------')


            

def main():
    device = 'cuda:0' if cuda else 'cpu'
    
    MODEL_DIR = os.path.join('weights', 'cpm_unet_epoch_1_best.pkl')
    
    model = CPM_UNet(num_stages=3, num_joints=17).to(device)
    model.load_state_dict(torch.load(MODEL_DIR))

    test(device, model)



if  __name__ == "__main__":
    main()