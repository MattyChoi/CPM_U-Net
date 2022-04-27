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
    # xs = np.array(xs[pts_nonzero])
    # ys = np.array(ys[pts_nonzero])
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



def crop_test(img, ele_anno, use_rotate=True, use_hflip=False, crop_size=256):

    bbox = ele_anno['bbox']
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

    #ys = np.array([ys[i]-bb_y1 for i in range(len(ys)) if i in pts_nonzero])
    bbox[0] -= bb_x1
    bbox[1] -= bb_y1
    

    cen[0] = int((bb_x2-bb_x1)/2)
    cen[1] = int((bb_y2-bb_y1)/2)


    # resize
    H,W = img.shape[0], img.shape[1]
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

    return img, cen


def crop_check(img, ele_anno, use_rotate=True, use_hflip=False, crop_size=256):
    # get bbox
    pts = ele_anno['landmarks']

    bbox = ele_anno['bbox']
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

    #ys = np.array([ys[i]-bb_y1 for i in range(len(ys)) if i in pts_nonzero])
    bbox[0] -= bb_x1
    bbox[1] -= bb_y1
    

    cen[0] = int((bb_x2-bb_x1)/2)
    cen[1] = int((bb_y2-bb_y1)/2)


    # resize
    H,W = img.shape[0], img.shape[1]
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

    return img, cen, pts


def offset_orig_coords(img_shape, hmap_size):
    '''
    Assume joints is shape 17 x 2 where first dimension is x and second dim is y
    '''
    offset_pair = np.zeros(2)
    height, width = img_shape
    if height > width:
        scale = height / hmap_size
        offset = (height - width) // 2
        offset_pair[0] = offset
    else:
        scale = width / hmap_size
        offset = (width - height) // 2
        offset_pair[1] = offset

    return offset_pair, scale


def show_heatmaps(img, heatmaps, c=np.zeros((2)), num_fig=1):
    '''
    :param img: np array (H,W,3)
    :param heatmaps: np array (H,W,num_pts)
    :param c: center, np array (2,)
    how to deal with negative in heatmaps ??? 
    '''
    H, W = img.shape[0], img.shape[1]
    dict_name = {
    0: 'Original Image',
    1: 'REye',
    2: 'LEye',
    3: 'Nose',
    4: 'Head',
    5: 'Neck',
    6: 'RShoulder',
    7: 'RElbow',
    8: 'RWrist',
    9: 'LShoulder',
    10: 'LElbow',
    11: 'LWrist',
    12: 'Hip',
    13: 'RKnee',
    14: 'RAnkle',
    15: 'LKnee',
    16: 'LAnkle',
    17: 'Tail'
    }

    # resize heatmap to size of image
    if heatmaps.shape[0] != H:
        heatmaps = skimage.transform.resize(heatmaps, (H, W))

    plt.figure(num_fig)
    for i in range(heatmaps.shape[2]):
        plt.subplot(4, 5, i + 1)
        plt.title(dict_name[i])
        if i == 0:
            plt.imshow(img)
        else:
            plt.imshow(heatmaps[:, :, i - 1])
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