import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np

import json
import os

from net import CPM_UNet
from cpm import CPM
from load_data import sanity_check_OMC
from utils import AverageMeter, show_heatmaps, offset_orig_coords

cuda = torch.cuda.is_available()

test_losses = AverageMeter()

def test(device, model):
    num_joints = 17

    model.eval()
    test_dataset = sanity_check_OMC()
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)

    for i, (img, cmap, img_shape, landmarks) in enumerate(test_loader):
        img = torch.FloatTensor(img).to(device)
        cmap = torch.FloatTensor(cmap).to(device)
        img_shape = img_shape[0][:2]
        landmarks = landmarks[0]

        pred_heatmaps = model(img, cmap)
        pred_hmap = pred_heatmaps[-1][0].cpu().detach().numpy().transpose((1,2,0))[:,:,:num_joints]
        offset, scale = offset_orig_coords(img_shape, pred_hmap.shape[0])

        pred_landmarks = []
        for joint_num in range(num_joints):
            pair = np.array(np.unravel_index(np.argmax(pred_hmap[:, :, joint_num]),
                                        img_shape))

            pair = pair * scale
            pair -= offset
            y, x = pair

            pred_landmarks.append(int(x))
            pred_landmarks.append(int(y))

        print(pred_landmarks)
        print(landmarks)
        print('--------------------------------------------------------')


            

def main():
    device = 'cuda:0' if cuda else 'cpu'
    
    MODEL_DIR = os.path.join('weights', 'cpm_epoch1_best.pkl')
    
    model = CPM(num_stages=3, num_joints=17).to(device)
    model.load_state_dict(torch.load(MODEL_DIR))

    test(device, model)



if  __name__ == "__main__":
    main()