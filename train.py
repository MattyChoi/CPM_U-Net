import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import numpy as np

from net import CPM_UNet
from load_data import OMC
from utils import AverageMeter, show_heatmaps, save_checkpoint

cuda = torch.cuda.is_available()

train_losses = AverageMeter()
test_losses = AverageMeter()

def train(device, optimizer, model, criterion):
    model.train()
    train_dataset = OMC(is_training=True)
    train_loader = torch_data.DataLoader(train_dataset, batch_size=16, shuffle=True, \
                                        collate_fn=train_dataset.collate_fn, num_workers=4)

    for img, heatmap, centermap in train_loader:
        img = torch.FloatTensor(img).to(device)
        ground_hmap = torch.FloatTensor(heatmap).to(device)
        centermap = torch.FloatTensor(centermap).to(device)

        pred_heatmaps = model(img, centermap)

        losses = [criterion(ground_hmap, pred) for pred in pred_heatmaps]

        loss = sum(losses)
        train_losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(device, model, criterion):
    model.eval()
    test_dataset = OMC(is_training=False)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                        collate_fn=test_dataset.collate_fn, num_workers=4)

    for i, (img, heatmap, centermap) in enumerate(test_loader):
        img = torch.FloatTensor(img).to(device)
        heatmap = torch.FloatTensor(heatmap).to(device)
        centermap = torch.FloatTensor(centermap).to(device)

        score1, score2, score3, score4, score5, score6 = model(img,centermap)

        loss1 = criterion(heatmap, score1)
        loss2 = criterion(heatmap, score2)
        loss3 = criterion(heatmap, score3)
        loss4 = criterion(heatmap, score4)
        loss5 = criterion(heatmap, score5)
        loss6 = criterion(heatmap, score6)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        test_losses.update(loss.item(), img.size(0))

        # show predicted heatmaps 
        if i == 0:
            heatmaps_pred_copy = score6[0]
            heatmaps_copy = heatmap[0] 
            img_copy = img[0]
            heatmaps_pred_np = heatmaps_pred_copy.detach().cpu().numpy()
            heatmaps_pred_np = np.transpose(heatmaps_pred_np, (1, 2, 0))
            heatmaps_np = heatmaps_copy.detach().cpu().numpy()
            heatmaps_np = np.transpose(heatmaps_np, (1, 2, 0))
            img_np = img_copy.detach().cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            
            show_heatmaps(img_np, heatmaps_pred_np, num_fig=1)
            show_heatmaps(img_np, heatmaps_np, num_fig=2)
            

def main():
    device = 'cuda:0' if cuda else 'cpu'
    
    # model.load_state_dict(torch.load(MODEL_DIR))
    model = CPM_UNet(num_stages=3, num_joints=17).to(device)

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
            path_ckpt = './weights/cpm_epoch_' + str(e+1)
            if test_losses.avg < best_test_loss:
                # save the model
                save_checkpoint(model.state_dict(), True, path_ckpt)
                print('===============CHECKPOINT PARAMS SAVED AT EPOCH %d ===============' %(e+1))
                best_test_loss = test_losses.avg
            else:
                save_checkpoint(model.state_dict(), False, path_ckpt)
                
            test_losses.reset()
    
    print(' Training Complete !')



if  __name__ == "__main__":
    main()