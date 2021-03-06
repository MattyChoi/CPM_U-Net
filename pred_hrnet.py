import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np

import json
import os
import hydra
# from omegaconf import DictConfig, OmegaConf

from models.hrnet import HRNet
from load_data import test_OMC_HRNET
from utils import AverageMeter, show_heatmaps, get_landmarks_from_preds

cuda = torch.cuda.is_available()

test_losses = AverageMeter()

def test(device, model, dir, store_pred_json=True):
    num_joints = 17

    model.eval()
    test_dataset = test_OMC_HRNET()
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)


    if store_pred_json:
        with open(dir) as f:
            dic = json.load(f)

    for i, (img, bbox) in enumerate(test_loader):
        img = torch.FloatTensor(img).to(device)
        bbox = bbox[0]

        pred_heatmaps = model(img)
        
        if store_pred_json:
            pred_hmap = pred_heatmaps[0].cpu().detach().numpy().transpose((1,2,0))[:,:,:num_joints]
            # show_heatmaps(img[0].cpu().detach().numpy().transpose((1,2,0)), pred_hmap)
            landmarks = get_landmarks_from_preds(pred_hmap, bbox, num_joints=num_joints)
            dic['data'][i]['landmarks'] = landmarks
            print(landmarks)
        
        if i % 1000 == 0: print("Iteration " + str(i))

    # dump into json file

    if store_pred_json:
        with open('test_prediction.json', 'w') as outfile:
            json.dump(dic, outfile)

            

@hydra.main(version_base=None, config_path="./models", config_name="config")
def main(params):
    device = 'cpu'
    
    MODEL_DIR = os.path.join('weights', 'hrnet_epoch_1_best.pkl')
    model = HRNet(params)
    model.load_state_dict(torch.load(MODEL_DIR))
    model = model.to(device)

    test_anno_dir = os.path.join('data', 'test_prediction.json')

    test(device, model, test_anno_dir, store_pred_json=True)



if  __name__ == "__main__":
    main()