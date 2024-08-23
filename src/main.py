import os
import tensorflow as tf

from argparse_utils import parse_args
from model import unet_3D_xy

# set cuda visable device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main(_):
    # load training parameter #
    param_sets = parse_args()
    param_set = param_sets[0]

    print('====== Phase >>> %s <<< ======' % param_set['phase'])

    if not os.path.exists(param_set['chkpoint_dir']):
        os.makedirs(param_set['chkpoint_dir'])
    if not os.path.exists(param_set['chkpoint_dir2']):
        os.makedirs(param_set['chkpoint_dir2'])
    if not os.path.exists(param_set['chkpoint_dir3']):
        os.makedirs(param_set['chkpoint_dir3'])
    if not os.path.exists(param_set['labeling_dir']):
        os.makedirs(param_set['labeling_dir'])

    # GPU setting, per_process_gpu_memory_fraction means 95% GPU MEM ,allow_growth means unfixed memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = unet_3D_xy(sess, param_set)

        if param_set['phase'] == 'train_step1' or param_set['phase'] == 'train_step3':
            model.train()
        if param_set['phase'] == 'train_step2_pred_gent':
            model.train_step2_pred_gent()
        if param_set['phase'] == 'train_step2':
            model.train_step2()
        elif param_set['phase'] == 'test':
            model.test()
            # model.test_generate_map()
        elif param_set['phase'] == 'crsv':
            model.test4crsv()

if __name__ == '__main__':
    tf.app.run()
