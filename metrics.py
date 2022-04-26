import os
import json
import numpy as np
import numpy.linalg as LA


def mpjpe(ground, preds, num_joints=17):
    error = np.zeros(num_joints)
    num_imgs = len(ground)
    
    for img in ground:
        width = ground[img]['bbox'][2]
        diff = ground[img]['landmarks'] - preds[img]['landmarks']
        error += LA.norm(diff, axis=1) / width

    return error / num_imgs


def pck(ground, preds, num_joints=17, tol=0.2):
    error = 0
    num_imgs = len(ground)
    
    for img in ground:
        width = ground[img]['bbox'][2]
        diff = ground[img]['landmarks'] - preds[img]['landmarks']
        error += (LA.norm(diff, axis=1) / width < tol).sum()

    return error / (num_imgs * num_joints)


def ap(ground, preds, num_joints=17, tol=0.5):
    error = 0
    num_imgs = len(ground)
    
    for img in ground:
        width = ground[img]['bbox'][2]
        error += (oks(ground[img]['landmarks'], preds[img]['landmarks'], width) >= tol).sum()

    return error / (num_imgs * num_joints)


def oks(truth, pred, width, k=np.ones(17)):
    diff = truth - pred
    return np.exp( ( np.sum(diff * diff, axis=1) / (width * k))**2 / -2.0) 


def json_to_dic(path):
    with open(path) as f:
        json_dic = json.load(f)
        dic = {}
        for item in json_dic['data']:
            landmarks = item['landmarks']
            pos_pairs = []

            for i in range(len(landmarks)//2):
                pos_pairs.append([landmarks[i], landmarks[i+1]])

            dic[item['file']] = {
                'bbox': item['bbox'],
                'landmarks': np.array(pos_pairs)
            }
    return dic


def main():
    path = r'C:\Users\matth\OneDrive\Documents\Storage\CSCI_5561\cpm-tf'
    ground_path = os.path.join(path, 'val_annotation.json')
    preds_path = os.path.join(path, 'val_preds.json')

    ground = json_to_dic(ground_path)
    preds = json_to_dic(preds_path)

    print(mpjpe(ground, preds))
    print(pck(ground, preds))
    print(ap(ground, preds))

if __name__ == '__main__':
    main()