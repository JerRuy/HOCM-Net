import numpy as np
import copy
import nibabel as nib

from scipy.ndimage import measurements
from scipy.spatial.distance import directed_hausdorff


from medpy import metric
import medpy.metric.binary as medpyMetrics

# Remove small connected components
def remove_minor_cc(vol_data, rej_ratio, rename_map):
    rem_vol = copy.deepcopy(vol_data)
    class_n = len(rename_map)
    # retrieve all classes
    for c in range(1, class_n):
        if c == 6:
            print( 'processing class %d...' % c)

            class_idx = (vol_data==rename_map[c])*1
            # print("1", np.shape(class_idx))
            class_vol = np.sum(class_idx)
            # print("2", class_vol )
            labeled_cc, num_cc = measurements.label(class_idx)
            # print('3',labeled_cc, num_cc )
            # retrieve all connected components in this class
            for cc in range(1, num_cc+1):
                single_cc = ((labeled_cc==cc)*1)
                single_vol = np.sum(single_cc)
                # remove if too small
                if single_vol / (class_vol*1.0) < rej_ratio:
                    rem_vol[labeled_cc==cc] = 0

    return rem_vol


# calculate evaluation metrics for segmentation
def seg_eval_metric(pred_label, gt_label):
    class_n = np.unique(gt_label)
    # dice
    dice_c = dice_n_class(move_img=pred_label, refer_img=gt_label)
    return dice_c

# dice value
def dice_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)
    print(c_list)
    dice_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c


# conformity value
def conform_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    conform_c = []
    for c in range(len(c_list)):
        # intersection
        ints = np.sum(((move_img == c_list[c]) * 1) * ((refer_img == c_list[c]) * 1))
        # sum
        sums = np.sum(((move_img == c_list[c]) * 1) + ((refer_img == c_list[c]) * 1)) + 0.0001
        # dice
        dice_temp = (2.0 * ints) / sums
        # conformity
        conform_temp = (3*dice_temp - 2) / dice_temp

        conform_c.append(conform_temp)

    return conform_c


# Jaccard index
def jaccard_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    jaccard_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # union
        uni = np.sum(np.logical_or(move_img_c, refer_img_c)*1) + 0.0001

        jaccard_c.append(ints / uni)

    return jaccard_c


# precision and recall
def precision_recall_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)

    precision_c = []
    recall_c = []
    for c in range(len(c_list)):
        move_img_c = (move_img == c_list[c])
        refer_img_c = (refer_img == c_list[c])
        # intersection
        ints = np.sum(np.logical_and(move_img_c, refer_img_c)*1)
        # precision
        prec = ints / (np.sum(move_img_c*1) + 0.001)
        # recall
        recall = ints / (np.sum(refer_img_c*1) + 0.001)

        precision_c.append(prec)
        recall_c.append(recall)

    return precision_c, recall_c

def Cal_HD95(pred, target):
    pred = pred#.cpu().numpy()
    target = target#.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    elif np.sum(pred) == 0 and np.count_nonzero(target):
        return 99
    else:
        # Edge cases that medpy cannot handle
        return -1

def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = Cal_HD95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dc, jc, hd, asd


def Cal_HD(pred, target):
    pred = pred  # .cpu().numpy()
    target = target  # .cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd = np.percentile(np.hstack((surDist1, surDist2)), 100)
        return hd
    elif np.sum(pred) == 0 and np.count_nonzero(target):
        return 99
    else:
        # Edge cases that medpy cannot handle
        return -1

