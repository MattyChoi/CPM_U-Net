import torch
import torch.utils.data as torch_data

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json

from net import CPM_UNet
from load_data import OMC
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR+FATAL

MPII_FILE_DIR = '../mpii_human_pose_v1'

cuda = torch.cuda.is_available()


def predict():
    """
    img: (H,W,3)
    """
    device = 'cuda:0' if cuda else 'cpu'
    
    # model.load_state_dict(torch.load(MODEL_DIR))
    model = CPM_UNet(num_stages=3, num_joints=17).to(device)
    MODEL_DIR = './weights/cpm_epoch_43_best.pkl'
    model.load_state_dict(torch.load(MODEL_DIR))
    model = model.to(device)

    model.eval()
    testset = OMC(is_training=False)

    eval_loader = torch_data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2, collate_fn=testset.collate_fn)
    for i, imgs in enumerate(eval_loader):
        
        imgs_torch = torch.FloatTensor(imgs).to(device)
        # centermap is just the center of image, i.e. (184,184)
        centermap = generate_heatmap(np.zeros((368,368)), pt=[184,184])
        centermap_torch = torch.FloatTensor(centermap).unsqueeze(0).unsqueeze(1).to(device) # add dim

        pred_hmaps = model(imgs_torch, centermap_torch)
        
        joints = heatmaps_to_coords(pred_hmaps[-1][0].cpu().detach().numpy().transpose((1,2,0))[:,:,:16] )




if __name__ == "__main__":
    predict()