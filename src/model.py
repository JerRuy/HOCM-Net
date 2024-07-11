from __future__ import division
import os
import time
from glob import glob
import cv2
import scipy.ndimage
from ops import *
from utils import *
from seg_eval import *


class unet_3D_xy(object):
    """ Implementation of 3D U-net"""

    def __init__(self, sess, param_set):
        self.sess = sess
        self.phase = param_set['phase']
        self.batch_size = param_set['batch_size']
        self.inputI_size = param_set['inputI_size']
        self.inputI_chn = param_set['inputI_chn']
        self.output_chn = param_set['outputI_size']
        self.output_chn = param_set['output_chn']
        self.resize_r = param_set['resize_r']
        self.traindata_dir = param_set['traindata_dir']
        self.chkpoint_dir = param_set['chkpoint_dir']
        self.chkpoint_dir2 = param_set['chkpoint_dir2']
        self.chkpoint_dir3 = param_set['chkpoint_dir3']
        self.lr = param_set['learning_rate']
        self.beta1 = param_set['beta1']
        self.epoch = param_set['epoch']
        self.model_name = param_set['model_name']
        self.save_intval = param_set['save_intval']
        self.testdata_dir = param_set['testdata_dir']
        self.labeling_dir = param_set['labeling_dir']
        self.testlabel_dir = param_set['testlabel_dir']
        self.predlabel_dir = param_set['predlabel_dir']
        self.ovlp_ita = param_set['ovlp_ita']
        self.pred_filter = param_set['pred_filter']

        self.rename_map = param_set['rename_map']
        self.rename_map = [int(s) for s in self.rename_map.split(',')]

        # build model graph
        self.build_model()

    # dice loss function
    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, self.output_chn)
        # print("input_gt shape is: ", input_gt.shape)
        # print("pred shape is: ", pred.shape)
        dice = 0
        for i in range(self.output_chn):
            inse = tf.reduce_mean(pred[:, :, :, :, i] * input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i] * pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2 * inse / (l + r)
        return -dice

    # class-weighted cross-entropy loss function
    def softmax_weighted_loss(self, logits, labels):
        """
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param labels: ground_truth
        :return: softmax-weifhted loss
        """
        gt = tf.one_hot(labels, self.output_chn)
        pred = logits
        softmaxpred = tf.nn.softmax(pred)
        loss = 0
        for i in range(self.output_chn):
            gti = gt[:, :, :, :, i]
            predi = softmaxpred[:, :, :, :, i]
            weighted = 1 - (tf.reduce_sum(gti) / tf.reduce_sum(gt))
            # print("class %d"%(i))
            # print(weighted)
            loss = loss + -tf.reduce_mean(weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)))
        return loss

    # build model graph
    def build_model(self):
        input_chn = 10 if self.phase == "train_step2" else 1
        # input
        self.input_I = tf.placeholder(dtype=tf.float32,
                                      shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size,
                                             input_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int32,
                                       shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_size],
                                       name='target')
        # probability for classes
        self.pred_prob, self.pred_label, self.aux0_prob, self.aux1_prob, self.aux2_prob = self.unet_3D_model(
            self.input_I)
        # ========= dice loss
        self.main_dice_loss = self.dice_loss_fun(self.pred_prob, self.input_gt)
        # auxiliary loss
        self.aux0_dice_loss = self.dice_loss_fun(self.aux0_prob, self.input_gt)
        self.aux1_dice_loss = self.dice_loss_fun(self.aux1_prob, self.input_gt)
        self.aux2_dice_loss = self.dice_loss_fun(self.aux2_prob, self.input_gt)
        #
        self.total_dice_loss = self.main_dice_loss + 0.2 * self.aux0_dice_loss + 0.4 * self.aux1_dice_loss + 0.8 * self.aux2_dice_loss

        # ========= class-weighted cross-entropy loss
        self.main_wght_loss = self.softmax_weighted_loss(self.pred_prob, self.input_gt)
        self.aux0_wght_loss = self.softmax_weighted_loss(self.aux0_prob, self.input_gt)
        self.aux1_wght_loss = self.softmax_weighted_loss(self.aux1_prob, self.input_gt)
        self.aux2_wght_loss = self.softmax_weighted_loss(self.aux2_prob, self.input_gt)
        self.total_wght_loss = self.main_wght_loss + 0.3 * self.aux0_wght_loss + 0.6 * self.aux1_wght_loss + 0.9 * self.aux2_wght_loss

        self.total_loss = 100.0 * self.total_dice_loss + self.total_wght_loss
        # self.total_loss = self.total_wght_loss

        # trainable variables
        self.u_vars = tf.trainable_variables()

        # extract the layers for fine tuning
        ft_layer = ['conv1/kernel:0',
                    'conv2/kernel:0',
                    'conv3a/kernel:0',
                    'conv3b/kernel:0',
                    'conv4a/kernel:0',
                    'conv4b/kernel:0']

        self.ft_vars = []
        for var in self.u_vars:
            for k in range(len(ft_layer)):
                if ft_layer[k] in var.name:
                    self.ft_vars.append(var)
                    break

        vars_pred_dict = dict()
        vars_dict = dict()
        for var in self.u_vars:
            key = var.name.split(':')[0]
            if '_0' in key:
                key = key.replace('_0/', '/', 1)
                vars_pred_dict[key] = var
            else:
                vars_dict[key] = var
        if vars_pred_dict:
            self.saver_pred = tf.train.Saver(var_list=vars_pred_dict)

        # create model saver
        self.saver = tf.train.Saver() if 'train' in self.phase else  tf.train.Saver(var_list=vars_dict)
        # saver to load pre-trained C3D model
        self.saver_ft = tf.train.Saver(self.ft_vars)

    # 3D unet graph
    def unet_3D_model(self, inputI):
        """3D U-net"""
        phase_flag = ('train' in self.phase)
        concat_dim = 4

        # down-sampling path
        # compute down-sample path in gpu0
        with tf.device("/gpu:0"):
            conv1_1, pool1 = conv_block(input=inputI, output_chn=64, phase_flag=phase_flag, conv3dName='conv1', batch_norm_scope="conv1_batch_norm", relu_name='conv1_relu', pool=True, pool_name='pool1')
            pool1_main, pool1_up, pool1_down = conv_pool(input=inputI, pool=pool1, poolc_output_chn=64, pool_output_chn=128, pool_size = 2, concat_dim = concat_dim, pyramid_name='pool_p1', concat_name='concat_p1',  poolc_name='conv1p', pool_up_deconv3d_name='deconv2-1', pool_down_pool_name='pool1_up')
            #main
            conv2_1_main, conv2_relu_main = conv_block(input=pool1_main, output_chn=128/2, phase_flag=phase_flag, conv3dName='conv2_main', batch_norm_scope="conv2_batch_norm_main", relu_name='conv2_relu')
            #up
            conv2_1_up, conv2_up = conv_block(input=pool1_up, output_chn=128/4, phase_flag=phase_flag, conv3dName='conv2_up', batch_norm_scope="conv2_batch_norm_up", relu_name='conv2_relu', pool=True, pool_name='pool2_up')
            #down
            conv2_1_down, conv2_relu_down = conv_block(input=pool1_down, output_chn=128/4, phase_flag=phase_flag, conv3dName='conv2_down', batch_norm_scope="conv2_batch_norm_down", relu_name='conv2_relu_down')
            conv2_down = Deconv3d(conv2_relu_down, 32, name='deconv2-2')

            conv2_1 = tf.concat([conv2_relu_main, conv2_down, conv2_up], axis=concat_dim, name='concat_2_2')
            conv2_1, pool2 = conv_block(input=conv2_1, output_chn=128, kernel_size=1, phase_flag=phase_flag, conv3dName='conv2_2', batch_norm_scope="conv2_batch_norm", relu_name='conv2_relu', pool=True, pool_name='pool2_main')


            # ###################################################################################################################################
            # ###################################################################################################################################
            # ##########################################################################################################
            # pyramid2
            pool2_main, pool2_up, pool2_down = conv_pool(input=inputI, pool=pool2, pool_output_chn=256, pool_size = 4, concat_dim = concat_dim, pyramid_name='pool_p2', concat_name='concat_p2',  poolc_name='conv2p', pool_up_deconv3d_name='deconv3-1', pool_down_pool_name='pool2_up')
            #main
            conv3_1_main, conv3_1_relu_main = conv_block(input=pool2_main, output_chn=256/2, phase_flag=phase_flag, conv3dName='conv3a_main', batch_norm_scope="conv3_1_batch_norm_main", relu_name='conv3_1_relu_main')
            conv3_2_main, conv3_2_relu_main = conv_block(input=conv3_1_relu_main, output_chn=256/2, phase_flag=phase_flag, conv3dName='conv3b_main', batch_norm_scope="conv3_2_batch_norm_main", relu_name='conv3_2_relu_main')
            # up
            conv3_1_up, conv3_up = conv_block(input=pool2_up, output_chn=256/4, phase_flag=phase_flag, conv3dName='conv3a_up', batch_norm_scope="conv3_1_batch_norm_up", relu_name='conv3_1_relu_up', pool=True, pool_name='pool3_up')
            #
            # down
            conv3_1_down, conv3_1_relu_down = conv_block(input=pool2_down, output_chn=256/4, phase_flag=phase_flag, conv3dName='conv3a_down', batch_norm_scope="conv3_1_batch_norm_down", relu_name='conv3_1_relu_down')
            conv3_2_down, conv3_2_relu_down = conv_block(input=conv3_1_relu_down, output_chn=256/4, phase_flag=phase_flag, conv3dName='conv3b_down', batch_norm_scope="conv3_2_batch_norm_down", relu_name='conv3_2_relu_down')
            conv3_down = Deconv3d(conv3_2_relu_down, 64, name='deconv3-2')
            #
            conv3_2 = tf.concat([conv3_2_relu_main, conv3_down, conv3_up], axis=concat_dim, name='concat_2_2')
            conv3_2, pool3 = conv_block(input=conv3_2, output_chn=256, phase_flag=phase_flag, conv3dName='conv3b', batch_norm_scope="conv3_2_batch_norm", relu_name='conv3_2_relu', pool=True, pool_name='pool3')
            #
            # ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            ############################################################################################################
            # pyramid3

            pool3_main, pool3_up, pool3_down = conv_pool(input=inputI, pool=pool3, pool_output_chn=512, pool_size = 8, concat_dim = concat_dim, pyramid_name='pool_p3', concat_name='concat_p4',  poolc_name='conv4p', pool_name_prifix='conv3p', pool_up_deconv3d_name='deconv4-1', pool_down_pool_name='pool3_up', pool_covn3d=False)

            ###################################################################################################################################
            conv4_1_main, conv4_1_relu_main = conv_block(input=pool3_main, output_chn=512/2, phase_flag=phase_flag, conv3dName='conv4a_main', batch_norm_scope="conv4_1_batch_norm_main", relu_name='conv4_1_relu_main')
            conv4_2_main, conv4_2_relu_main = conv_block(input=conv4_1_relu_main, output_chn=512/2, phase_flag=phase_flag, conv3dName='conv4b_main', batch_norm_scope="conv4_2_batch_norm_main", relu_name='conv4_2_relu_main')
            #up
            conv4_1_up, conv4_1_relu_up = conv_block(input=pool3_up, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4a_up', batch_norm_scope="conv4_1_batch_norm_up", relu_name='conv4_1_relu_up')
            conv4_2_up, conv4_up = conv_block(input=conv4_1_relu_up, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4b_up', batch_norm_scope="conv4_2_batch_norm_up", relu_name='conv4_2_relu_up', pool=True, pool_name='pool3_up')
            #down
            conv4_1_down, conv4_1_relu_down = conv_block(input=pool3_down, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4a_down', batch_norm_scope="conv4_1_batch_norm_down", relu_name='conv4_1_relu_down')
            conv4_2_down, conv4_2_relu_down = conv_block(input=conv4_1_relu_down, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4b_down', batch_norm_scope="conv4_2_batch_norm_down", relu_name='conv4_2_relu_down')
            conv4_down = Deconv3d(conv4_2_relu_down, 128, name='deconv4-2_down')
            #
            ###################################################################################################################################
            conv4_2 = tf.concat([conv4_2_relu_main, conv4_down, conv4_up], axis=concat_dim, name='concat_3_2')
            conv4_2, pool4 = conv_block(input=conv4_2, output_chn=512, phase_flag=phase_flag, conv3dName='conv4b', batch_norm_scope="conv4_2_batch_norm", relu_name='conv4_2_relu', pool=True, pool_name='pool4')
            #
            # ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################

            ############################################################################
            # pyramid4
            pool4_main, pool4_up, pool4_down = conv_pool(input=inputI, pool=pool4, pool_output_chn=512, pool_size = 16, concat_dim = concat_dim, pyramid_name='pool_p4', concat_name='concat_p5',  poolc_name='conv5p', pool_up_deconv3d_name='deconv5-1', pool_down_pool_name='pool5_up')

            conv5_1_main = conv_bn_relu(input=pool4_main, output_chn=512/2, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_1_main')
            conv5_2_main = conv_bn_relu(input=conv5_1_main, output_chn=512/2, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_2_main')

            conv5_1_up = conv_bn_relu(input=pool4_up, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_1_up')
            conv5_2_up = conv_bn_relu(input=conv5_1_up, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_2_up')
            conv5_up = tf.layers.max_pooling3d(inputs=conv5_2_up, pool_size=2, strides=2, name='pool5_up')

            conv5_1_down = conv_bn_relu(input=pool4_down, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_1_down')
            conv5_2_down = conv_bn_relu(input=conv5_1_down, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_2_down')
            conv5_down = Deconv3d(conv5_2_down, 128, name='deconv5-2_down')
            #
            conv5_2 = tf.concat([ conv5_2_main , conv5_up, conv5_down], axis=concat_dim, name='concat_5_2')
            _, conv5_2 = conv_block(input=conv5_2, output_chn=512, kernel_size=1, phase_flag=phase_flag, conv3dName='conv5_2', batch_norm_scope="conv5_2_batch_norm", relu_name='conv5_2_relu')

        # up-sampling path
        # compute up-sample path in gpu1
        with tf.device("/gpu:0"):
            deconv1_2 = deconv_block(conv5_2, 512, conv4_2, concat_dim, phase_flag, '1')
            deconv2_2 = deconv_block(deconv1_2, 256, conv3_2, concat_dim, phase_flag, '2')
            deconv3_2 = deconv_block(deconv2_2, 128, conv2_1, concat_dim, phase_flag, '3')
            deconv4_2 = deconv_block(deconv3_2, 64, conv1_1, concat_dim, phase_flag, '4')

            # predicted probability
            pred_prob = conv3d(input=deconv4_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True,
                               name='pred_prob')
            # ======================
            # auxiliary prediction 0
            aux0_prob = aux_prob(input=deconv1_2, output_chn=self.output_chn, num=2, name_prefix='0')
            # auxiliary prediction 1
            aux1_prob = aux_prob(input=deconv2_2, output_chn=self.output_chn, num=1, name_prefix='1')
            # auxiliary prediction 2
            aux2_prob = aux_prob(input=deconv3_2, output_chn=self.output_chn, num=0, name_prefix='2')
        with tf.device("/cpu:0"):
            # predicted labels
            soft_prob = tf.nn.softmax(pred_prob, name='pred_soft')
            pred_label = tf.argmax(soft_prob, axis=4, name='argmax')
            if self.phase == 'train_step1' or 'train_step2' in self.phase:
                return pred_prob, pred_label, aux0_prob, aux1_prob, aux2_prob

            ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################ ###################################################################################################################################
            # ###################################################################################################################################
            # ###################################################################################################################################
            #####################################################################################################################################
            ############################################################################################################
        with tf.device("/gpu:0"):
            soft_prob_new = filter_tensor(soft_prob, self.pred_filter)
            inputI_0 = tf.concat([inputI, soft_prob_new], axis=concat_dim, name='concat_p1_0')

            conv1_1_0, pool1_0 = conv_block(input=inputI_0, output_chn=64, phase_flag=phase_flag, conv3dName='conv1_0', batch_norm_scope="conv1_batch_norm_0", relu_name='conv1_relu', pool=True, pool_name='pool1_0')
            pool1_main, pool1_up_0, pool1_down = conv_pool(input=inputI_0, pool=pool1_0, poolc_output_chn=64, pool_output_chn=128, pool_size = 2, concat_dim = concat_dim, pyramid_name='pool_p1', concat_name='concat_p1',  poolc_name='conv1p_0', pool_up_deconv3d_name='deconv2-1_0', pool_down_pool_name='pool1_up_0')
            _, conv2_relu_main_0 = conv_block(input=pool1_main, output_chn=128/2, phase_flag=phase_flag, conv3dName='conv2_main_0', batch_norm_scope="conv2_batch_norm_main_0", relu_name='conv2_relu')
            _, conv2_up_0 = conv_block(input=pool1_up_0, output_chn=128/4, phase_flag=phase_flag, conv3dName='conv2_up_0', batch_norm_scope="conv2_batch_norm_up_0", relu_name='conv2_relu', pool=True, pool_name='pool2_up_0')
            _, conv2_relu_down_0 = conv_block(input=pool1_down, output_chn=128/4, phase_flag=phase_flag, conv3dName='conv2_down_0', batch_norm_scope="conv2_batch_norm_down_0", relu_name='conv2_relu_down_0')
            conv2_down_0 = Deconv3d(conv2_relu_down_0, 32, name='deconv2-2_0')
            conv2_1_0 = tf.concat([conv2_relu_main_0, conv2_down_0, conv2_up_0], axis=concat_dim, name='concat_2_0')
            conv2_1_0, pool2_0 = conv_block(input=conv2_1_0, output_chn=128, kernel_size=1, phase_flag=phase_flag, conv3dName='conv2_2_0', batch_norm_scope="conv2_batch_norm_0", relu_name='conv2_relu', pool=True, pool_name='pool2_main_0')
            pool2_main_0, pool2_up_0, pool2_down_0 = conv_pool(input=inputI_0, pool=pool2_0, pool_output_chn=256, pool_size = 4, concat_dim = concat_dim, pyramid_name='pool_p2', concat_name='concat_p2',  poolc_name='conv2p_0', pool_up_deconv3d_name='deconv3-1_0', pool_down_pool_name='pool2_up_0')
            _, conv3_1_relu_main_0 = conv_block(input=pool2_main_0, output_chn=256/2, phase_flag=phase_flag, conv3dName='conv3a_main_0', batch_norm_scope="conv3_1_batch_norm_main_0", relu_name='conv3_1_relu_main_0')
            _, conv3_2_relu_main_0 = conv_block(input=conv3_1_relu_main_0, output_chn=256/2, phase_flag=phase_flag, conv3dName='conv3b_main_0', batch_norm_scope="conv3_2_batch_norm_main_0", relu_name='conv3_2_relu_main_0')
            _, conv3_up_0 = conv_block(input=pool2_up_0, output_chn=256/4, phase_flag=phase_flag, conv3dName='conv3a_up_0', batch_norm_scope="conv3_1_batch_norm_up_0", relu_name='conv3_1_relu_up', pool=True, pool_name='pool3_up_0')
            _, conv3_1_relu_down_0 = conv_block(input=pool2_down_0, output_chn=256/4, phase_flag=phase_flag, conv3dName='conv3a_down_0', batch_norm_scope="conv3_1_batch_norm_down_0", relu_name='conv3_1_relu_down_0')
            _, conv3_2_relu_down_0 = conv_block(input=conv3_1_relu_down_0, output_chn=256/4, phase_flag=phase_flag, conv3dName='conv3b_down_0', batch_norm_scope="conv3_2_batch_norm_down_0", relu_name='conv3_2_relu_down_0')
            conv3_down_0 = Deconv3d(conv3_2_relu_down_0, 64, name='deconv3-2_0')
            conv3_2_0 = tf.concat([conv3_2_relu_main_0, conv3_down_0, conv3_up_0], axis=concat_dim, name='concat_2_0')
            conv3_2_0, pool3_0 = conv_block(input=conv3_2_0, output_chn=256, phase_flag=phase_flag, conv3dName='conv3b_0', batch_norm_scope="conv3_2_batch_norm_0", relu_name='conv3_2_relu', pool=True, pool_name='pool3_0')
            pool3_main_0, pool3_up_0, pool3_down_0 = conv_pool(input=inputI_0, pool=pool3_0, pool_output_chn=512, pool_size = 8, concat_dim = concat_dim, pyramid_name='pool_p3', concat_name='concat_p4',  poolc_name='conv4p_0', pool_name_prifix='conv3p_0', pool_up_deconv3d_name='deconv4-1_0', pool_down_pool_name='pool3_up_0', pool_covn3d=False)
            _, conv4_1_relu_main_0 = conv_block(input=pool3_main_0, output_chn=512/2, phase_flag=phase_flag, conv3dName='conv4a_main_0', batch_norm_scope="conv4_1_batch_norm_main_0", relu_name='conv4_1_relu_main_0')
            _, conv4_2_relu_main_0 = conv_block(input=conv4_1_relu_main_0, output_chn=512/2, phase_flag=phase_flag, conv3dName='conv4b_main_0', batch_norm_scope="conv4_2_batch_norm_main_0", relu_name='conv4_2_relu_main_0')
            _, conv4_1_relu_up_0 = conv_block(input=pool3_up_0, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4a_up_0', batch_norm_scope="conv4_1_batch_norm_up_0", relu_name='conv4_1_relu_up_0')
            _, conv4_up_0 = conv_block(input=conv4_1_relu_up_0, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4b_up_0', batch_norm_scope="conv4_2_batch_norm_up_0", relu_name='conv4_2_relu_up', pool=True, pool_name='pool3_up_0')
            _, conv4_1_relu_down_0 = conv_block(input=pool3_down_0, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4a_down_0', batch_norm_scope="conv4_1_batch_norm_down_0", relu_name='conv4_1_relu_down_0')

            _, conv4_2_relu_down_0 = conv_block(input=conv4_1_relu_down_0, output_chn=512/4, phase_flag=phase_flag, conv3dName='conv4b_down_0', batch_norm_scope="conv4_2_batch_norm_down_0", relu_name='conv4_2_relu_down_0')
            conv4_down_0 = Deconv3d(conv4_2_relu_down_0, 128, name='deconv4-2_down_0')
            conv4_2_0 = tf.concat([conv4_2_relu_main_0, conv4_down_0, conv4_up_0], axis=concat_dim, name='concat_3_0')
            conv4_2_0, pool4_0 = conv_block(input=conv4_2_0, output_chn=512, phase_flag=phase_flag, conv3dName='conv4b_0', batch_norm_scope="conv4_2_batch_norm_0", relu_name='conv4_2_relu', pool=True, pool_name='pool4_0')
            pool4_main_0, pool4_up_0, pool4_down_0 = conv_pool(input=inputI_0, pool=pool4_0, pool_output_chn=512, pool_size = 16, concat_dim = concat_dim, pyramid_name='pool_p4', concat_name='concat_p5',  poolc_name='conv5p_0', pool_up_deconv3d_name='deconv5-1_0', pool_down_pool_name='pool5_up')

            conv5_1_main_0 = conv_bn_relu(input=pool4_main_0, output_chn=512/2, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_1_main_0')
            conv5_2_main_0 = conv_bn_relu(input=conv5_1_main_0, output_chn=512/2, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_2_main_0')


            conv5_1_up_0 = conv_bn_relu(input=pool4_up_0, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_1_up_0')
            conv5_2_up_0 = conv_bn_relu(input=conv5_1_up_0, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_2_up_0')
            conv5_up_0 = tf.layers.max_pooling3d(inputs=conv5_2_up_0, pool_size=2, strides=2, name='pool5_up')


            conv5_1_down_0 = conv_bn_relu(input=pool4_down_0, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_1_down_0')
            conv5_2_down_0 = conv_bn_relu(input=conv5_1_down_0, output_chn=512/4, kernel_size=3, stride=1, use_bias=False,
                                   is_training=phase_flag, name='conv5_2_down_0')
            conv5_down_0 = Deconv3d(conv5_2_down_0, 128, name='deconv5-2_down_0')
            conv5_2_0 = tf.concat([ conv5_2_main_0 , conv5_up_0, conv5_down_0], axis=concat_dim, name='concat_5_0')
            _, conv5_2_0 = conv_block(input=conv5_2_0, output_chn=512, kernel_size=1, phase_flag=phase_flag, conv3dName='conv5_2_0', batch_norm_scope="conv5_2_batch_norm_0", relu_name='conv5_2_relu')
        with tf.device("/gpu:0"):
            deconv1_2_0 = deconv_block(conv5_2_0, 512, conv4_2_0, concat_dim, phase_flag, '1_0')
            deconv2_2_0 = deconv_block(deconv1_2_0, 256, conv3_2_0, concat_dim, phase_flag, '2_0')
            deconv3_2_0 = deconv_block(deconv2_2_0, 128, conv2_1_0, concat_dim, phase_flag, '3_0')
            deconv4_2_0 = deconv_block(deconv3_2_0, 64, conv1_1_0, concat_dim, phase_flag, '4_0')
             # predicted probability
            pred_prob_0 = conv3d(input=deconv4_2_0, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True,
                               name='pred_prob_0')
            aux0_prob_0 = aux_prob(input=deconv1_2_0, output_chn=self.output_chn, num=2, name_prefix='0_0')
            # auxiliary prediction 1
            aux1_prob_0 = aux_prob(input=deconv2_2_0, output_chn=self.output_chn, num=1, name_prefix='1_0')
            # auxiliary prediction 2
            aux2_prob_0 = aux_prob(input=deconv3_2_0, output_chn=self.output_chn, num=0, name_prefix='2_0')
        with tf.device("/cpu:0"):
            # predicted labels
            soft_prob_0 = tf.nn.softmax(pred_prob_0, name='pred_soft')
            pred_label_0 = tf.argmax(soft_prob_0, axis=4, name='argmax')
        return pred_prob_0, pred_label_0, aux0_prob_0, aux1_prob_0, aux2_prob_0

    # train step1 step3 function
    def train(self):
        """Train 3D U-net"""
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.total_loss,
                                                                                               var_list=self.u_vars)
        # initialization
        init_op = tf.global_variables_initializer()
        # print(init_op)
        self.sess.run(init_op)

        # load C3D model
        # self.initialize_finetune()

        # save .log
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        # if self.load_chkpoint(self.chkpoint_dir):
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        # load all volume files
        pair_list = glob('{}/*.nii.gz'.format(self.traindata_dir))
        pair_list.sort()
        img_clec, label_clec = load_data_pairs(pair_list, self.resize_r, self.rename_map)
        # temporary file to save loss
        loss_log = open("loss.txt", "w")

        for epoch in np.arange(self.epoch):
            start_time = time.time()
            # train batch
            batch_img, batch_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1,
                                                       flip_flag=True, rot_flag=True)
            # validation batch
            batch_val_img, batch_val_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size,
                                                               chn=1, flip_flag=True, rot_flag=True)

            # Update 3D U-net
            _, cur_train_loss = self.sess.run([u_optimizer, self.total_loss],
                                              feed_dict={self.input_I: batch_img, self.input_gt: batch_label})
            # self.log_writer.add_summary(summary_str, counter)

            # current and validation loss
            cur_valid_loss = self.total_loss.eval({self.input_I: batch_val_img, self.input_gt: batch_val_label})
            cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: batch_val_img})
            # dice_loss = self.sess.run(self.dice_loss, feed_dict={self.input_I: batch_val_img, self.input_gt: batch_val_label})
            print(np.unique(batch_label))
            print(np.unique(cube_label))
            # dice value
            dice_c = []
            for c in range(self.output_chn):
                ints = np.sum(((batch_val_label[0, :, :, :] == c) * 1) * ((cube_label[0, :, :, :] == c) * 1))
                union = np.sum(((batch_val_label[0, :, :, :] == c) * 1) + ((cube_label[0, :, :, :] == c) * 1)) + 0.0001
                dice_c.append((2.0 * ints) / union)
            print(dice_c)

            loss_log.write("%s    %s\n" % (cur_train_loss, cur_valid_loss))

            counter += 1
            print("Epoch: [%2d] time: %4.4f, train_loss: %.8f, valid_loss: %.8f" % (
            epoch, time.time() - start_time, cur_train_loss, cur_valid_loss))

            chkpoint_dir = self.chkpoint_dir if self.phase == "train_step1" else self.chkpoint_dir2
            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(chkpoint_dir, self.model_name, counter)
                

    # train step2 predicted data file generate
    def train_step2_pred_gent(self):
        """train step2 3D U-net"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list of train dataset
        train_list = glob('{}/*image.nii.gz'.format(self.traindata_dir))
        train_list.sort()

        for k in range(0, len(train_list)):
            # load the volume
            print('loading ', train_list[k])
            train_name = train_list[k].split('/')[-1]
            vol_file = nib.load(train_list[k])
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()

            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')

            print(np.array(vol_data.shape), resize_dim)
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn,
                                           self.ovlp_ita)
            # predict on each cube
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp

                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm})
                cube_label_list.append(cube_label)
            # compose cubes into a volume
            composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.inputI_size, self.ovlp_ita,
                                                   self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            # rename label
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')
            print(np.unique(composed_label))

            # save predicted label
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            labeling_path = os.path.join(self.predlabel_dir, train_name)
            labeling_vol = nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, labeling_path)

    # train step2 function
    def train_step2(self):
        """Train 3D U-net"""
        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.total_loss,
                                                                                               var_list=self.u_vars)
        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load C3D model
        # self.initialize_finetune()

        # save .log
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        # load all volume files
        pair_list = glob('{}/*.nii.gz'.format(self.traindata_dir))
        pair_list.sort()
        img_clec, label_clec = load_data_pairs(pair_list, self.resize_r, self.rename_map)

        pred_list = glob('{}/*.nii.gz'.format(self.predlabel_dir))
        pred_list.sort()
        pred_clec = load_pred_data(pred_list, self.resize_r, self.rename_map)

        # temporary file to save loss
        loss_log = open("loss.txt", "w")

        for epoch in np.arange(self.epoch):
            start_time = time.time()
            # train batch
            # batch_img, batch_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1,
            #                                            flip_flag=True, rot_flag=True)

            batch_img, batch_label = get_batch_patches_fuse(img_clec, label_clec, pred_clec, self.inputI_size, self.batch_size, self.pred_filter, chn=1,
                                                       flip_flag=True, rot_flag=True)


            # validation batch
            batch_val_img, batch_val_label = get_batch_patches_fuse(img_clec, label_clec, pred_clec, self.inputI_size, self.batch_size, self.pred_filter, 
                                                               chn=1, flip_flag=True, rot_flag=True)

            # Update 3D U-net
            _, cur_train_loss = self.sess.run([u_optimizer, self.total_loss],
                                              feed_dict={self.input_I: batch_img, self.input_gt: batch_label})

            # current and validation loss
            cur_valid_loss = self.total_loss.eval({self.input_I: batch_val_img, self.input_gt: batch_val_label})
            cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: batch_val_img})
            print(np.unique(batch_label))
            print(np.unique(cube_label))

            # dice value
            dice_c = []
            for c in range(self.output_chn):
                ints = np.sum(((batch_val_label[0, :, :, :] == c) * 1) * ((cube_label[0, :, :, :] == c) * 1))
                union = np.sum(((batch_val_label[0, :, :, :] == c) * 1) + ((cube_label[0, :, :, :] == c) * 1)) + 0.0001
                dice_c.append((2.0 * ints) / union)
            print(dice_c)

            loss_log.write("%s    %s\n" % (cur_train_loss, cur_valid_loss))

            counter += 1
            print("Epoch: [%2d] time: %4.4f, train_loss: %.8f, valid_loss: %.8f" % (
            epoch, time.time() - start_time, cur_train_loss, cur_valid_loss))

            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir3, self.model_name, counter)

        loss_log.close()

    # test the model
    def test(self):
        """Test 3D U-net"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # print(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir, self.chkpoint_dir2):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list of testing dataset
        test_list = glob('{}/*.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        # test
        for k in range(0, len(test_list)):
            # load the volume
            print('loading ', test_list[k])
            # print('loading ',test_list[k].split('/')[9])
            test_name = test_list[k].split('/')[-1]
            vol_file = nib.load(test_list[k])
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()

            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')

            print(np.array(vol_data.shape), resize_dim)
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # decompose volume into list of cubes
            # print(np.shape(vol_data_resz))
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn,
                                           self.ovlp_ita)
            # predict on each cube
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp

                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.inputI_size, self.ovlp_ita,
                                                   self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            # rename label
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')
            print(np.unique(composed_label))

            # for s in range(composed_label.shape[2]):
            #     cv2.imshow('volume_seg', np.concatenate(((vol_data_resz[:, :, s]*255.0).astype('uint8'), (composed_label[:, :, s]/4).astype('uint8')), axis=1))
            #     cv2.waitKey(30)

            # save predicted label
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            # remove minor connected components
            # composed_label_resz = remove_minor_cc(composed_label_resz, rej_ratio=0.3, rename_map=self.rename_map)

            labeling_path = os.path.join(self.labeling_dir, test_name)
            # labeling_path = os.path.join(self.labeling_dir, ('ct_test_' + str(2001+k) + '_label.nii.gz'))
            labeling_vol = nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, labeling_path)

    # test function for cross validation
    def test4crsv(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list of testing dataset
        test_list = glob('{}/*image.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        # all dice
        all_dice = np.zeros([int(len(test_list) / 2), self.output_chn])

        # test
        for k in range(0, len(test_list), 1):
            # load the volume
            vol_file = nib.load(test_list[k])
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()
            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')

            # print(np.array(vol_data.shape),resize_dim)
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn,
                                           self.ovlp_ita)
            # predict on each cube
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp

                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.inputI_size, self.ovlp_ita,
                                                   self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            # rename label
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')
            print(np.unique(composed_label))

            # for s in range(composed_label.shape[2]):
            #     cv2.imshow('volume_seg', np.concatenate(((vol_data_resz[:, :, s]*255.0).astype('uint8'), (composed_label[:, :, s]/4).astype('uint8')), axis=1))
            #     cv2.waitKey(30)

            dir_path, file_name = os.path.split(test_list[k])
            label_file_name = file_name.replace("image", "label")

            # save predicted label
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            labeling_path = os.path.join(self.labeling_dir, label_file_name)
            labeling_vol = nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, labeling_path)

            # evaluation
            testlabel_path = os.path.join(self.testlabel_dir, label_file_name)
            gt_file = nib.load(testlabel_path)
            gt_label = gt_file.get_data().copy()

            k_dice_c = seg_eval_metric(composed_label_resz, gt_label)
            print("dice_c:", k_dice_c)

            #            dc, jc, hd, asd= calculate_metric_percase(composed_label_resz, gt_label)
            #            print(' dc:',dc,' jc:',jc,' 95HD:',hd,' ASD:',asd)

            print("predice path: ", labeling_path)
            print("label   path: ", testlabel_path)

            predict_one = composed_label_resz
            label_one = gt_label

            predict_one[predict_one != 6] = 0
            label_one[label_one != 6] = 0
            predict_one[predict_one == 6] = 1
            label_one[label_one == 6] = 1

            dc, jc, hd, asd = calculate_metric_percase(predict_one, label_one)
            print(' dc:', dc, ' jc:', jc, ' 95HD:', hd, ' ASD:', asd)

            all_dice[int(k / 2), :] = np.asarray(k_dice_c)

        mean_dice = np.mean(all_dice, axis=0)
        print("average dice: ")
        print(mean_dice)

    # test the model
    def test_generate_map(self):
        """Test 3D U-net"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list of testing dataset
        test_list = glob('{}/*.nii.gz'.format(self.testdata_dir))
        test_list.sort()

        # test
        test_cnt = 0
        for k in range(0, len(test_list), 2):
            print("processing # %d volume..." % k)
            # load the volume
            vol_file = nib.load(test_list[k])
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()
            resize_dim = (np.array(vol_data.shape) * self.resize_r).astype('int')
            #
            # print(np.array(vol_data.shape),resize_dim)
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn,
                                           self.ovlp_ita)
            print("=== cube list is built!")
            # predict on each cube
            cube_prob_list = []
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp
                # probability cube
                cube_probs = self.sess.run(self.pred_prob, feed_dict={self.input_I: cube2test_norm})
                cube_prob_list.append(cube_probs)
                # label cube
                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_prob_orig = compose_prob_cube2vol(cube_prob_list, resize_dim, self.inputI_size, self.ovlp_ita,
                                                       self.output_chn)
            print("=== prob volume is composed!")

            # composed_label_orig = np.argmax(composed_prob_orig, axis=3)

            # resize probability map and save
            # composed_prob_resz = np.array([vol_data.shape[0], vol_data.shape[1], vol_data.shape[2], self.output_chn])
            min_prob = np.min(composed_prob_orig)
            max_prob = np.max(composed_prob_orig)
            for p in range(self.output_chn):
                composed_prob_resz = resize(composed_prob_orig[:, :, :, p], vol_data.shape, order=1,
                                            preserve_range=True)

                # composed_prob_resz = (composed_prob_resz - np.min(composed_prob_resz)) / (np.max(composed_prob_resz) - np.min(composed_prob_resz))
                composed_prob_resz = (composed_prob_resz - min_prob) / (max_prob - min_prob)
                composed_prob_resz = composed_prob_resz * 255
                composed_prob_resz = composed_prob_resz.astype('int16')

                c_map_path = os.path.join(self.labeling_dir, ('auxi_' + str(2001 + k) + '_c' + str(p) + '.nii.gz'))
                c_map_vol = nib.Nifti1Image(composed_prob_resz, ref_affine)
                nib.save(c_map_vol, c_map_path)
                print("=== probl volume is saved!")
                # c_map_path = os.path.join("/home/xinyang/project_xy/mmwhs2017/dataset/common/dice_map", ('auxi_' + str(test_cnt) + '_c' + str(p) + '.npy'))
                # np.save(c_map_path, composed_prob_resz)

            test_cnt = test_cnt + 1

    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s_%s" % (self.batch_size, self.output_chn)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir, checkpoint_dir2=''):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.batch_size, self.output_chn)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("ckpt_name is : ", ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            if len(checkpoint_dir2) != 0:
                checkpoint_dir2 = os.path.join(checkpoint_dir2, model_dir)
                ckpt2 = tf.train.get_checkpoint_state(checkpoint_dir2)
                ckpt_name2 = os.path.basename(ckpt2.model_checkpoint_path)
                self.saver_pred.restore(self.sess, os.path.join(checkpoint_dir2, ckpt_name2))
            return True
        else:
            return False

    # load C3D model
    def initialize_finetune(self):
        checkpoint_dir = '../outcome/model/C3D_unet_1chn'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_ft.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

