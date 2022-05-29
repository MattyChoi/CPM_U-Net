import torch
import torch.nn as nn
import torch.utils.data as torch_data
import os

from models.hrnet import HRNet
from load_data import OMC_HRNET
from utils import AverageMeter, show_heatmaps, save_checkpoint
from tqdm import tqdm
import hydra
# from omegaconf import DictConfig, OmegaConf

cuda = torch.cuda.is_available()

train_losses = AverageMeter()
test_losses = AverageMeter()


def train(device, optimizer, model, criterion):
    model.train()
    train_dataset = OMC_HRNET(is_training=True)
    train_loader = torch_data.DataLoader(train_dataset, batch_size=8, shuffle=True, \
                                        collate_fn=train_dataset.collate_fn, num_workers=1)

    with tqdm(train_loader, unit="batch") as batch:
        for img, hmap in batch:
            img = torch.FloatTensor(img).to(device)
            hmap = torch.FloatTensor(hmap).to(device)

            pred_heatmaps = model(img)

            loss = criterion(hmap, pred_heatmaps)
            train_losses.update(loss.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(device, model, criterion):
    model.eval()
    test_dataset = OMC_HRNET(is_training=False)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)

    for img, hmap in test_loader:
        img = torch.FloatTensor(img).to(device)
        hmap = torch.FloatTensor(hmap).to(device)

        pred_heatmaps = model(img)
        loss = criterion(hmap, pred_heatmaps)

        test_losses.update(loss.item(), img.size(0))
            

@hydra.main(version_base=None, config_path="./models", config_name="config")
def main(params):
    device = 'cuda:0' if cuda else 'cpu'
    
    # model.load_state_dict(torch.load(MODEL_DIR))
    model = HRNet(params)
    model.init_weights()
    model = model.to(device)

    epoches = 50   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction='mean')

    best_test_loss = 100.0
    print('==================== START TRAINING ====================')
    for e in range(epoches):
        train(device, optimizer, model, criterion)

        print('Epoch: {} || Training Loss: {}'.format(e+1, train_losses.avg))
        train_losses.reset()

        if (e+1) % 1 == 0:
            test(device, model, criterion)
            print('Epoch: {} || Testing Loss: {}'.format(e+1, test_losses.avg))
            pack_ckpt = os.path.join('weights', 'hrnet_epoch_' + str(e+1))
            if test_losses.avg < best_test_loss:
                # save the model
                save_checkpoint(model.state_dict(), True, pack_ckpt)
                print('===============CHECKPOINT PARAMS SAVED AT EPOCH %d ===============' %(e+1))
                best_test_loss = test_losses.avg
            else:
                save_checkpoint(model.state_dict(), False, pack_ckpt)
                
            test_losses.reset()
    
    print(' Training Complete !')



if  __name__ == "__main__":
    main()
