import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.layers import UpSampling3D

#######################
# 3d functions
#######################
# convolution
def conv3d(input, output_chn, kernel_size, stride, use_bias=False, name='conv'):
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding='same', data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)


def conv_bn_relu(input, output_chn, kernel_size, stride, use_bias, is_training, name):
    with tf.variable_scope(name):
        conv = conv3d(input, output_chn, kernel_size, stride, use_bias, name='conv')
        # with tf.device("/cpu:0"):
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu


# deconvolution
def Deconv3d(input, output_chn, name):
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
    filter = tf.get_variable(name+"/filter", shape=[4, 4, 4, output_chn, in_channels], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0, 0.01), regularizer=slim.l2_regularizer(0.0005))

    conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, in_depth * 2, in_height * 2, in_width * 2, output_chn],
                                  strides=[1, 2, 2, 2, 1], padding="SAME", name=name)
    return conv


def Upsampling3d(input, output_chn, name):
    batch, in_depth, in_height, in_width, in_channels = [int(d) for d in input.get_shape()]
    # filter = tf.get_variable(name+"/filter", shape=[4, 4, 4, output_chn, in_channels], dtype=tf.float32,
    #                          initializer=tf.random_normal_initializer(0, 0.01), regularizer=slim.l2_regularizer(0.0005))

    # conv = tf.nn.conv3d_transpose(value=input, filter=filter, output_shape=[batch, in_depth * 2, in_height * 2, in_width * 2, output_chn],
    #                               strides=[1, 2, 2, 2, 1], padding="SAME", name=name)
    output_tensor = tf.nn.interpolate3d(input,
                                        size=[in_depth * 2, in_height * 2, in_width * 2],
                                        method='linear')
    return output_tensor




def deconv_bn_relu(input, output_chn, is_training, name):
    with tf.variable_scope(name):
        conv = Deconv3d(input, output_chn, name='deconv')
        # with tf.device("/cpu:0"):
        bn = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu

def conv_bn_relu_x3(input, output_chn, kernel_size, stride, use_bias, is_training, name):
    with tf.variable_scope(name):
        z=conv_bn_relu(input, output_chn, kernel_size, stride, use_bias, is_training, "dense1")
        z_out = conv_bn_relu(z, output_chn, kernel_size, stride, use_bias, is_training, "dense2")
        z_out = conv_bn_relu(z_out, output_chn, kernel_size, stride, use_bias, is_training, "dense3")
        return z+z_out

def conv_block(input, output_chn, phase_flag, conv3dName, batch_norm_scope, relu_name, kernel_size=3, pool=False, pool_name=''):
    conv = conv3d(input=input, output_chn=output_chn, kernel_size=kernel_size, stride=1, use_bias=False, name=conv3dName)
    bn = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope=batch_norm_scope)
    relu = tf.nn.relu(bn, name=relu_name)
    if pool:
        pool = tf.layers.max_pooling3d(inputs=relu, pool_size=2, strides=2, name=pool_name)
        return conv, pool
    return conv, relu

def conv_pool(input, pool,  pool_size, pool_output_chn, concat_dim, pyramid_name, concat_name,  poolc_name,  pool_up_deconv3d_name, pool_down_pool_name, poolc_output_chn=0, pool_name_prifix='', pool_covn3d=True):
    postfix = ''
    if poolc_output_chn==0:
        poolc_output_chn = pool_output_chn
    if pool_name_prifix == '':
        pool_name_prifix = poolc_name
    if '_0' in pool_name_prifix:
        pool_name_prifix = pool_name_prifix.split('_0')[0]
        postfix='_0'

    pyramid1 = tf.layers.max_pooling3d(inputs=input, pool_size=pool_size, strides=pool_size, name=pyramid_name)  #pool_p1
    pool1c = tf.concat([pyramid1, pool], axis=concat_dim, name=concat_name)
    if pool_covn3d==True:
        pool1c = conv3d(input=pool1c, output_chn=poolc_output_chn, kernel_size=1, stride=1, use_bias=False,name=poolc_name)
    else:
        conv3d(input=pool1c, output_chn=poolc_output_chn, kernel_size=1, stride=1, use_bias=False,name=poolc_name)
    
    pool1_main = conv3d(input=pool1c, output_chn=pool_output_chn/2, kernel_size=1, stride=1, use_bias=False, name=pool_name_prifix+'_main'+postfix)

    pool1_up = conv3d(input=pool1c, output_chn=pool_output_chn/4, kernel_size=1, stride=1, use_bias=False, name=pool_name_prifix+'_up'+postfix)
    pool1_up = Deconv3d(pool1_up, pool_output_chn/4, name=pool_up_deconv3d_name)

    pool1_down = conv3d(input=pool1c, output_chn=pool_output_chn/4, kernel_size=1, stride=1, use_bias=False, name=pool_name_prifix+'_down'+postfix)
    pool1_down = tf.layers.max_pooling3d(inputs=pool1_down, pool_size=2, strides=2, name=pool_down_pool_name)
    return pool1_main, pool1_up, pool1_down

def deconv_block(input, output_chn, concat_input, concat_dim, phase_flag, name_prefix):
    postfix=''
    if '_0' in name_prefix:
        name_prefix = name_prefix.split('_0')[0]
        postfix='_0'
    deconv = deconv_bn_relu(input=input, output_chn=output_chn, is_training=phase_flag, name='deconv'+name_prefix+"_1" + postfix)
    concat = tf.concat([deconv, concat_input], axis=concat_dim, name='concat_' + name_prefix)
    conv = conv_bn_relu(input=concat, output_chn=output_chn // 2, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='deconv'+name_prefix+'_2'+postfix)
    return conv


def aux_prob(input, output_chn, num, name_prefix):
    postfix=''
    if '_0' in name_prefix:
        name_prefix = name_prefix.split('_0')[0]
        postfix='_0'
    aux_conv = conv3d(input=input, output_chn=output_chn, kernel_size=1, stride=1, use_bias=True, name='aux'+name_prefix+'_conv' + postfix)
    index = 0
    while(num > index):
        index = index + 1
        aux_conv = Deconv3d(input=aux_conv, output_chn=output_chn, name='aux'+name_prefix+'_deconv_' + str(index)+postfix)
    aux_prob = Deconv3d(input=aux_conv, output_chn=output_chn, name='aux'+name_prefix+'_prob' + postfix)
    return aux_prob


def filter_tensor(tensor, values):
    values = list(map(int, values.split(','))) 
    masks = [tf.equal(tensor, value) for value in values] 
    combined_mask = tf.reduce_any(tf.stack(masks), axis=0) 
    filtered_tensor = tf.where(combined_mask, tensor, tf.zeros_like(tensor))
    return filtered_tensor