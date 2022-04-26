import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import skimage.transform
import skimage.filters
import torch
import shutil


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0.
		self.avg = 0.
		self.sum = 0.
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, filename + '_latest.pkl')
    if is_best:
        shutil.copyfile(filename + '_latest.pkl', filename + '_best.pkl')


def load_img(dir):
    img = mpimg.imread(dir)
    return img


def generate_heatmap(heatmap, pt, sigma_valu=2):
    '''
    :param heatmap: should be a np zeros array with shape (H,W) (only 1 channel), not (H,W,1)
    :param pt: point coords, np array
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: value for gaussian blur
    :return: a np array of one joint heatmap with shape (H,W)
    This function is obsolete, use 'generate_heatmaps()' instead.
    '''
    heatmap[int(pt[1])][int(pt[0])] = 1
    heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)
    am = np.max(heatmap)
    heatmap = heatmap/am
    return heatmap



def generate_heatmaps(img, pts, sigma_valu=2):
    '''
    Generate 16 heatmaps
    :param img: np arrray img, (H,W,C)
    :param pts: joint points coords, np array, same resolu as img
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: vaalue for gaussian blur
    :return: np array heatmaps, (H,W,num_pts)
    '''
    H, W = img.shape[0], img.shape[1]
    num_pts = len(pts)
    heatmaps = np.zeros((H, W, num_pts + 1))
    for i, pt in enumerate(pts):
        # Filter unavailable heatmaps
        if pt[0] == 0 and pt[1] == 0:
            continue
        # Filter some points out of the image
        if pt[0] >= W:
            pt[0] = W-1
        if pt[1] >= H:
            pt[1] = H-1
        heatmap = heatmaps[:, :, i]
        heatmap[int(pt[1])][int(pt[0])] = 1  # reverse sequence
        heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)  ##(H,W,1) -> (H,W)
        am = np.max(heatmap)
        heatmap = heatmap / am  # scale to [0,1]
        heatmaps[:, :, i] = heatmap

    heatmaps[:, :, num_pts] = 1.0 - np.max(heatmaps[:, :, :num_pts], axis=2) # add background dim

    return heatmaps



def crop(img, ele_anno, use_rotate=True, use_hflip=False, crop_size=256):

    # get bbox
    pts = ele_anno['landmarks']
    # print("pts", pts)
    
    
    # pts = np.array([[x, y] for x, y in zip(xs, ys)])

    bbox = ele_anno['bbox']
    vis = np.array(ele_anno['visibility'])
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    pts_nonzero = np.where(vis != 0)
    pts_zero = np.where(vis == 0)
    xs = np.array(pts[0::2]).T
    ys = np.array(pts[1::2]).T
    xs = np.array(xs[pts_nonzero])
    ys = np.array(ys[pts_nonzero])
    # print("ptsx", xs)
    # print("ptsy", ys)
    cen = np.array((1,2))
    cen[0] = int(bbox[0] + bbox[2]/2)
    cen[1] = int(bbox[1] + bbox[3]/2)

    H,W = img.shape[0], img.shape[1]
    # topleft:x1,y1  bottomright:x2,y2
    bb_x1 = int(bbox[0])
    bb_y1 = int(bbox[1])
    bb_x2 = int(bbox[0] + bbox[2])
    bb_y2 = int(bbox[1] + bbox[3])

    newX = bb_x2-bb_x1
    newY = bb_y2-bb_y1
    if(newX>newY):
        dif = newX-newY
        bb_y1-=int(dif/2)
        bb_y2+=int(dif/2)
    else:
        dif=newY-newX
        bb_x1-=int(dif/2)
        bb_x2+=int(dif/2)

    if bb_x1<0 or bb_x2>W or bb_y1<0 or bb_y2>H:
        pad = int(max(-bb_x1, bb_x2-W, -bb_y1, bb_y2-H))
        img = np.pad(img, ((pad, pad),(pad,pad),(0,0)), 'constant')
    else:
        pad = 0
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show() 
    img = img[bb_y1+pad:bb_y2+pad, bb_x1+pad:bb_x2+pad]

    xs = np.where(xs != 0, xs-bb_x1, xs)
    ys = np.where(ys != 0, ys-bb_y1, ys)
    #ys = np.array([ys[i]-bb_y1 for i in range(len(ys)) if i in pts_nonzero])
    bbox[0] -= bb_x1
    bbox[1] -= bb_y1
    
    cen[0] = int((bb_x2-bb_x1)/2)
    cen[1] = int((bb_y2-bb_y1)/2)

    # resize
    H,W = img.shape[0], img.shape[1]
    xs = xs*crop_size/W
    ys = ys*crop_size/H
    cen[0] = cen[0]*crop_size/W
    cen[1] = cen[1]*crop_size/H
    # print("scale", scale)
    # print("bbox", bbox)

    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    img = cv2.resize(img, (crop_size, crop_size))
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()

    pts = [[xs[i], ys[i]] for i in range(len(xs))]

    return img, pts, cen



def change_resolu(img, pts, c, resolu_out):
    '''
    :param img: np array of the origin image
    :param pts: joint points np array corresponding to the image, same resolu as img
    :param c: center
    :param resolu_out: a list or tuple
    :return: img_out, pts_out, c_out under resolu_out
    '''
    H_in = img.shape[0]
    W_in = img.shape[1]
    H_out = resolu_out[0]
    W_out = resolu_out[1]
    H_scale = H_in/H_out
    W_scale = W_in/W_out

    pts_out = pts/np.array([W_scale, H_scale, 1])
    c_out = c/np.array([W_scale, H_scale])
    img_out = skimage.transform.resize(img, resolu_out)

    return img_out, pts_out, c_out


# TODO: Modify flaws in showing joints
# Chest and nest often overlap
def show_stack_joints(img, pts, c=[0, 0], draw_lines=True, num_fig=1):
    '''
    Not support batch 
    :param img: np array, (H,W,C)
    :param pts: same resolu as img, joint points, np array (16,3) or (16,2)
    :param c: center, np array (2,)
    '''
    # In case pts is not np array
    pts = np.array(pts)
    dict_style = {

        0: 'origin img',

        1: ['left ankle', 'b', 'x'],
        2: ['left knee', 'b', '^'],
        3: ['left hip', 'b', 'o'],

        4: ['right hip', 'r', 'o'],
        5: ['right knee', 'r', '^'],
        6: ['right ankle', 'r', 'x'],

        7: ['belly', 'y', 'o'],
        8: ['chest', 'y', 'o'],
        9: ['neck', 'y', 'o'],
        10: ['head', 'y', '*'],

        11: ['left wrist', 'b', 'x'],
        12: ['left elbow', 'b', '^'],
        13: ['left shoulder', 'b', 'o'],

        14: ['right shoulder', 'r', 'o'],
        15: ['right elbow', 'r', '^'],
        16: ['right wrist', 'r', 'x']
    }
    plt.figure(num_fig)
    plt.imshow(img)
    list_pt_H, list_pt_W = [], []
    list_pt_cH, list_pt_cW = [], []
    for i in range(pts.shape[0]):
        list_pt_W.append(pts[i, 0])  # x axis
        list_pt_H.append(pts[i, 1])  # y axis
    list_pt_cW.append(c[0])
    list_pt_cH.append(c[1])
    for i in range(pts.shape[0]):
        plt.scatter(list_pt_W[i], list_pt_H[i], color=dict_style[i+1][1], marker=dict_style[i+1][2])
    plt.scatter(list_pt_cW, list_pt_cH, color='b', marker='*')
    if draw_lines:                                
        # Body
        plt.plot(list_pt_W[6:10], list_pt_H[6:10], color='y', linewidth=2)
        plt.plot(list_pt_W[2:4], list_pt_H[2:4], color='y', linewidth=2)
        plt.plot(list_pt_W[12:14], list_pt_H[12:14], color='y', linewidth=2)
        # Left arm
        plt.plot(list_pt_W[10:13], list_pt_H[10:13], color='b', linewidth=2)
        # Right arm
        plt.plot(list_pt_W[13:16], list_pt_H[13:16], color='r', linewidth=2)
        # Left leg
        # change: if left ankle or knee doesn't exist
        if pts[0,1] != 0:
            if pts[1,1] != 0:
                plt.plot(list_pt_W[0:3], list_pt_H[0:3], color='b', linewidth=2)
            else:
                pass  # werid condition
        else:
            if pts[1,1] != 0:
                plt.plot(list_pt_W[1:3], list_pt_H[1:3], color='b', linewidth=2)
            else:
                pass  # draw nothing

        # Right leg
        if pts[5,1] != 0:
            if pts[4,1] != 0:
                plt.plot(list_pt_W[3:6], list_pt_H[3:6], color='r', linewidth=2)
            else:
                pass # werid condition
        else:
            if pts[4,1] != 0:
                plt.plot(list_pt_W[3:5], list_pt_H[3:5], color='r', linewidth=2)
            else:
                pass  # draw nothing
    plt.axis('off')
    plt.show()

    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.margins(0,0)
    # plt.savefig('./imgs/preprocess_%d.jpg' %num_fig,bbox_inches='tight',pad_inches=0.0) # remove padding




def show_heatmaps(img, heatmaps, c=np.zeros((2)), num_fig=1):
    '''
    :param img: np array (H,W,3)
    :param heatmaps: np array (H,W,num_pts)
    :param c: center, np array (2,)
    how to deal with negative in heatmaps ??? 
    '''
    H, W = img.shape[0], img.shape[1]
    dict_name = {
        0: 'origin img',
        1: 'left ankle',
        2: 'left knee',
        3: 'left hip',
        4: 'right hip',
        5: 'right knee',
        6: 'right ankle',
        7: 'belly',
        8: 'chest',
        9: 'neck',
        10: 'head',
        11: 'left wrist',
        12: 'left elbow',
        13: 'left shoulder',
        14: 'right shoulder',
        15: 'right elbow',
        16: 'right wrist',
        17: 'background'
    }

    # resize heatmap to size of image
    if heatmaps.shape[0] != H:
        heatmaps = skimage.transform.resize(heatmaps, (H, W))

    heatmap_c = generate_heatmap(np.zeros((H, W)), c)
    plt.figure(num_fig)
    for i in range(heatmaps.shape[2] + 1):
        plt.subplot(4, 5, i + 1)
        plt.title(dict_name[i])
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(heatmaps[:, :, i - 1])
        plt.axis('off')
    plt.subplot(4, 5, 20)
    plt.imshow(heatmap_c)  # Only take in (H,W) or (H,W,3)
    plt.axis('off')
    plt.show()



def heatmaps_to_coords(heatmaps, resolu_out=(368,368), prob_threshold=0.2):
    '''
    :param heatmaps: (46,46,16)
    :param resolu_out: output resolution list
    :return coord_joints: np array, shape (16,2)
    '''

    num_joints = heatmaps.shape[2]
    # Resize
    heatmaps = cv2.resize(heatmaps, resolu_out)
    print('heatmaps.SHAPE', heatmaps.shape)

    coord_joints = np.zeros((num_joints, 2))
    for i in range(num_joints):
        heatmap = heatmaps[..., i]
        max = np.max(heatmap)
        # Only keep points larger than a threshold
        if max > prob_threshold:
            idx = np.where(heatmap == max)
            H = idx[0][0]
            W = idx[1][0]
        else:
            H = 0
            W = 0
        coord_joints[i] = [W, H]
    return coord_joints