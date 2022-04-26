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


# for generating center map
def gen_cmap(heatmap, pt, sigma_valu=2):
    '''
    :param heatmap: should be a np zeros array with shape (H,W) (only 1 channel), not (H,W,1)
    :param pt: point coords, np array
    :param sigma: should be a tuple with odd values (obsolete)
    :param sigma_valu: value for gaussian blur
    :return: a np array of one joint heatmap with shape (H,W)
    '''
    heatmap[int(pt[1])][int(pt[0])] = 1
    heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)
    am = np.max(heatmap)
    heatmap = heatmap/am
    return heatmap


# for generating ground truth heatmaps
def gen_hmaps(img, pts, sigma_valu=2):
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
        if pt[0] == 0 and pt[1] == 0: continue

        # Filter some points out of the image
        if pt[0] >= W: pt[0] = W-1
        if pt[1] >= H: pt[1] = H-1
        heatmap = heatmaps[:, :, i]
        heatmap[int(pt[1])][int(pt[0])] = 1  # reverse sequence
        heatmap = skimage.filters.gaussian(heatmap, sigma=sigma_valu)  ##(H,W,1) -> (H,W)
        am = np.max(heatmap)
        heatmap = heatmap / am  # scale to [0,1]
        heatmaps[:, :, i] = heatmap

    heatmaps[:, :, num_pts] = 1.0 - np.max(heatmaps[:, :, :num_pts], axis=2) # add background dim

    return heatmaps



def crop(img, ele_anno, use_rotate=True, use_hflip=False, crop_size=368):
    # get bbox
    pts = ele_anno['joint_self']
    cen = ele_anno['objpos'].copy()

    pts = np.array(pts)
    pts_nonzero = np.where(pts[:,1] != 0)[0]
    pts_zero = np.where(pts[:,1] == 0)[0]
    xs = pts[:, 0]
    #xs = pts[pts_nonzero][:, 0]
    ys = pts[:, 1]
    #ys = pts[pts_nonzero][:, 1]
    bbox = [(max(max(xs[pts_nonzero]),cen[0]) + min(min(xs[pts_nonzero]), cen[0]) )/2.0,
            (max(max(ys[pts_nonzero]),cen[1]) + min(min(ys[pts_nonzero]), cen[1]) )/2.0,
            (max(max(xs[pts_nonzero]),cen[0]) - min(min(xs[pts_nonzero]), cen[0]) )*1.3,
            (max(max(ys[pts_nonzero]),cen[1]) - min(min(ys[pts_nonzero]), cen[1]) )*1.3]
    bbox = np.array(bbox)



    H,W = img.shape[0], img.shape[1]
    scale = np.random.uniform(0.8, 1.3) # given data
    bbox[2] *= scale
    bbox[3] *= scale
    # topleft:x1,y1  bottomright:x2,y2
    bb_x1 = int(bbox[0] - bbox[2]/2)
    bb_y1 = int(bbox[1] - bbox[3]/2)
    bb_x2 = int(bbox[0] + bbox[2]/2)
    bb_y2 = int(bbox[1] + bbox[3]/2)

    if bb_x1<0 or bb_x2>W or bb_y1<0 or bb_y2>H:
        pad = int(max(-bb_x1, bb_x2-W, -bb_y1, bb_y2-H))
        img = np.pad(img, ((pad, pad),(pad,pad),(0,0)), 'constant')
    else:
        pad = 0
    img = img[bb_y1+pad:bb_y2+pad, bb_x1+pad:bb_x2+pad]

    xs = np.where(xs != 0, xs-bb_x1, xs)
    ys = np.where(ys != 0, ys-bb_y1, ys)
    #ys = np.array([ys[i]-bb_y1 for i in range(len(ys)) if i in pts_nonzero])
    bbox[0] -= bb_x1
    bbox[1] -= bb_y1
    
    cen[0] -= bb_x1
    cen[1] -= bb_y1

    # resize
    H,W = img.shape[0], img.shape[1]
    xs = xs*crop_size/W
    ys = ys*crop_size/H
    cen[0] = cen[0]*crop_size/W
    cen[1] = cen[1]*crop_size/H
    img = cv2.resize(img, (crop_size, crop_size))

    pts = [[xs[i], ys[i], pts[i,2]] for i in range(len(xs))]

    return img, pts, cen




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

    heatmap_c = gen_hmaps(np.zeros((H, W)), c)
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



def hmaps_to_coords(heatmaps):
    num_joints = heatmaps.shape[2] - 1
    coord_joints = np.zeros((num_joints, 2))

    for joint_num in range(num_joints):
        y, x = np.unravel_index(np.argmax(heatmaps[:, :, joint_num]),
                                (heatmaps.shape[0], heatmaps.shape[1]))
        coord_joints[joint_num] = [x, y]

    return coord_joints



def visualize_result(test_img, FLAGS, stage_heatmap_np):
    last_heatmap = stage_heatmap_np[-1][0, :, :, 0: FLAGS.joints].reshape(
        (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))

    joint_coord_set = np.zeros((FLAGS.joints, 2))
    
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    for joint_num in range(FLAGS.joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                        (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num % 6 ]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=5, color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num % 6]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=5, color=joint_color, thickness=-1)
        print(joint_coord, joint_num)

    return test_img