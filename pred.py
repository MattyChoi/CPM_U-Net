import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np

import json
import os

from models.cpm import CPM
from load_data import test_OMC
from utils import AverageMeter, show_heatmaps, offset_orig_coords

cuda = torch.cuda.is_available()

test_losses = AverageMeter()

def test(device, model, dir):
    num_joints = 17

    model.eval()
    test_dataset = test_OMC()
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)

    with open(dir) as f:
        dic = json.load(f)

    for i, (img, cmap, bbox) in enumerate(test_loader):
        img = torch.FloatTensor(img).to(device)
        cmap = torch.FloatTensor(cmap).to(device)
        bbox = bbox[0]
        resize_shape = (bbox[3], bbox[2])

        pred_heatmaps = model(img, cmap)
        pred_hmap = pred_heatmaps[-1][0].cpu().detach().numpy().transpose((1,2,0))[:,:,:num_joints]
        offset, scale = offset_orig_coords(resize_shape, pred_hmap.shape[0])

        landmarks = []
        for joint_num in range(num_joints):
            pair = np.array(np.unravel_index(np.argmax(pred_hmap[:, :, joint_num]),
                                        pred_hmap.shape[:2]))

            pair = pair * scale
            pair -= offset
            y, x = pair

            landmarks.append(int(x))
            landmarks.append(int(y))

        dic['data'][i]['landmarks'] = landmarks
        
        if i % 1000 == 0: print("Iteration " + str(i))

    # dump into json file
    with open('test_annotation.json', 'w') as outfile:
        json.dump(dic, outfile)

            

def main():
    device = 'cuda:0' if cuda else 'cpu'
    
    MODEL_DIR = os.path.join('weights', 'cpm_epoch_1_best.pkl')
    
    model = CPM(num_stages=3, num_joints=17).to(device)
    model.load_state_dict(torch.load(MODEL_DIR))

    test_anno_dir = os.path.join('data', 'test_prediction.json')

    test(device, model, test_anno_dir)



if  __name__ == "__main__":
    main()