import numpy as np
import copy
import nibabel as nib

from scipy.ndimage import measurements
from scipy.spatial.distance import directed_hausdorff


from medpy import metric
import medpy.metric.binary as medpyMetrics

# Remove small connected components
def remove_minor_cc(vol_data, rej_ratio, rename_map):
    """Remove small connected components refer to rejection ratio"""
    """Usage
        # rename_map = [0, 205, 420, 500, 550, 600, 820, 850]
        # nii_path = '/home/xinyang/project_xy/mmwhs2017/dataset/ct_output/test/test_4.nii'
        # vol_file = nib.load(nii_path)
        # vol_data = vol_file.get_data().copy()
        # ref_affine = vol_file.affine
        # rem_vol = remove_minor_cc(vol_data, rej_ratio=0.2, class_n=8, rename_map=rename_map)
        # # save
        # rem_path = 'rem_cc.nii'
        # rem_vol_file = nib.Nifti1Image(rem_vol, ref_affine)
        # nib.save(rem_vol_file, rem_path)

        #===# possible be parallel in future
    """

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
    # # conformity
    # conform_c = conform_n_class(move_img=pred_label, refer_img=gt_label)
    # # jaccard
    # jaccard_c = jaccard_n_class(move_img=pred_label, refer_img=gt_label)
    # # precision and recall
    # precision_c, recall_c = precision_recall_n_class(move_img=pred_label, refer_img=gt_label)

    # return dice_c, conform_c, jaccard_c, precision_c, recall_c
    return dice_c

# dice value
def dice_n_class(move_img, refer_img):
    # list of classes
    c_list = np.unique(refer_img)
    # c_list = [0,1,2,3,4,5,6,7,8]
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
        # surDist1 = medpyMetrics.__surface_distances(pred, target, voxelspacing = [1.17, 1.17, 3])
        # surDist2 = medpyMetrics.__surface_distances(target, pred, voxelspacing = [1.17, 1.17, 3])
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



if __name__ == '__main__':
    # listing = [53,55,56,61,62, 63,64,65,66,67,69, 71,73,74,75,76,79]
    listing = [11,13,14,15,16,17,18,21,22]
    # listing = [11,12,13,21,22,23,24,25,27,31,32,34,35,36,38,41,42,43,44,45,46,47,48]
    # listing = [69,71,73,75,76,79]

    for i in range(len(listing)):
        # iter = 62
        iter =listing[i]
        # a_nii_path = '/home/zhangjw2/code/pytorch/8dheart/HCM_GD_resize/test/label/' + str(iter) + '_label.nii.gz'
        # b_nii_path = '/home/zhangjw2/code/pytorch/8dheart/hcmgd5/ds_ft_hybrid_4ct/result/'+str(iter)+'_image.nii.gz'

        # a_nii_path = '/home/zhangjw2/code/pytorch/8dheart/HCM_GD_resize/test1/label/' + str(iter) + '_label.nii.gz'
        # b_nii_path = '/home/zhangjw2/code/pytorch/8dheart/hcmgd3/ds_ft_hybrid_4ct/result-test1/'+str(iter)+'_image.nii.gz'

        # a_nii_path = '/home/zhangjw2/code/pytorch/8dheart/HCM_GD_resize/train/label/' + str(iter) + '_label.nii.gz'
        # b_nii_path = '/home/zhangjw2/code/pytorch/8dheart/hcmgd3/ds_ft_hybrid_4ct/result_train/'+str(iter)+'_image.nii.gz'

    #    a_nii_path = '/home/zhangjw2/code/pytorch/8dheart/HCM_GD_resize/test2/label/' + str(iter) + '_label.nii.gz'
    #    b_nii_path = '/home/zhangjw2/code/pytorch/8dheart/hcmgd5/ds_ft_hybrid_4ct/result-test2/'+str(iter)+'_image.nii.gz'

        a_nii_path = '/home/zlwu/code-heart-eswa/8dheart/HCM_GD_resize/test2/label/' + str(iter) + '_label.nii.gz'
        b_nii_path = '/home/zlwu/code-heart-eswa/8dheart/hcmgd5/ds_ft_hybrid_4ct/result-test2/'+str(iter)+'_image.nii.gz'

        print(a_nii_path)
        a_nii = nib.load(a_nii_path).get_data().copy()
        b_nii = nib.load(b_nii_path).get_data().copy()

        # from utils import *

        # b_nii = remove_minor_cc(b_nii, rej_ratio=0.3, rename_map=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        # print(np.shape(a_nii))
        # print(np.shape(b_nii))
        # dice_c, conform_c, jaccard_c, precision_c, recall_c = seg_eval_metric(a_nii, b_nii)

        dice_c = dice_n_class(a_nii, b_nii)
        print("dice_c is : ", dice_c)
        if len(dice_c) != 9:
            continue
        else:
            if i == 0: dice_total = np.array(dice_c)
            else: dice_total = dice_total + np.array(dice_c)
        # conform_c = conform_n_class(a_nii, b_nii)
        # jaccard_c = jaccard_n_class(a_nii, b_nii)
        # precision_c, recall_c = precision_recall_n_class(a_nii, b_nii)
        #
        # print(conform_c)
        # print(jaccard_c)
        # print(precision_c)
        # print(recall_c)
        predict_one = b_nii
        label_one   = a_nii

        predict_one[predict_one!=6]=0
        label_one[label_one!=6]=0

        predict_one[predict_one==6]=1
        label_one[label_one==6]=1



        # print(np.shape(predict_one))
        # print(np.shape(label_one))

        # hd_i = Cal_HD(predict_one, label_one)
        #
        # print( "HD:", hd_i)
        # hd95_i = Cal_HD95(predict_one, label_one)
        # print(' 95HD:',hd95_i)

        dc, jc, hd, asd= calculate_metric_percase(predict_one, label_one)

        print(' dc:',dc,' jc:',jc,' 95HD:',hd,' ASD:',asd)


    print(dice_total)
    print(dice_total/[i,i,i, i,i,i, i,i,i])
