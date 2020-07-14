import os
import sys
import shutil
import random
import scipy
import tensorflow as tf
from datetime import datetime
import progressbar
from image_data_handler_joint_multimodal import ImageDataHandler
from resnet18 import ResNet
from layer_blocks import *
from tensorflow.data import Iterator
from utils import *
import pickle as pickle
# numpy 1.16.1 ALLOW PICKLE=TRUE di default
from imgaug import imgaug as ia
from imgaug import augmenters as iaa

from keras.objectives import categorical_crossentropy
from keras.optimizers import RMSprop
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.metrics import categorical_accuracy
from keras import backend as K
from cmc_functions import *

from tensorflow.python.client import device_lib

import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
import cv2

print(device_lib.list_local_devices())

""" Configuration setting """

tf.set_random_seed(7)

# Data-related params
dataset_root_dir = "/content/drive/My Drive/Dataset/finale/"
params_root_dir = "/content/drive/My Drive/ParametriOggetti/resnet18_ocid_params"

dataset_train_dir_rgb = dataset_root_dir + ''
dataset_val_dir_rgb = dataset_root_dir + ''
params_dir_rgb = params_root_dir + '/resnet18_ocid_rgb++_params.npy'

params_dir_depth = params_root_dir + '/resnet18_ocid_surfnorm++_params.npy'

# todo occhio ai file
train_file = dataset_root_dir + '/train50.txt'
val_file = dataset_root_dir + '/val50.txt'
test_file = dataset_root_dir + '/test50.txt'

# Log params
tensorboard_log = '/content/tmp/tensorflow/'

# Solver params
learning_rate = [[0.001]]
num_epochs = 50 # 50
batch_size = [[64]]
num_neurons = [[100]]
l2_factor = [[0.0]]
maximum_norm = [[4]]
dropout_rate = [[0.4]]

delta = 0.05            #delta for early stopping
patience = 3            #patience for early stopping

depth_transf = [[256]]
transf_block = transformation_block_v1

# Checkpoint dir
checkpoint_dir = "/tmp/my_caffenet/"
if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)

# Input/Output
num_classes = 50
img_size = [224, 224]
num_channels = 3

simpleLog = "/content/drive/My Drive/RCFusionGPU/log.txt"

""" Online data augmentation """


def random_choice(start, end, _multiply):
    start = start * _multiply
    end = end * _multiply
    num = random.randrange(start, end + 1, 1)
    # print ("il num e'",num/_multiply)
    return float(num / _multiply)


def x_y_random_image(image):
    width = image.shape[1]
    hight = image.shape[0]

    border_x = int((256 * 5) / 100.0)
    border_y = int((256 * 9) / 100.0)

    pos_x = random.randrange(0 + border_x, width - border_x, 1)
    pos_y = random.randrange(0 + border_y, hight - border_y, 1)
    # print ("la pos e' ", pos_x , pos_y)
    return pos_x, pos_y


def data_aug(batch, batch_depth):
    num_img = batch.shape[0]
    list = []
    list_depth = []
    for i in range(num_img):
        val_fliplr = random.randrange(0, 2, 1)  # in questo modo il due non e compreso e restituisce i valori 0 o 1
        list.extend([iaa.Fliplr(val_fliplr)])
        list_depth.extend([iaa.Fliplr(val_fliplr)])

        val_fliplr = random.randrange(0, 2, 1)  # in questo modo il due non e compreso e restituisce i valori 0 o 1
        list.extend([iaa.Flipud(val_fliplr)])
        list_depth.extend([iaa.Flipud(val_fliplr)])

        val_scala = random.randrange(5, 11, 1)
        val = float(val_scala / 10.0)
        list.extend([iaa.Affine(val, mode='edge')])
        list.extend([iaa.Affine(10.0 / val_scala, mode='edge')])
        list_depth.extend([iaa.Affine(val, mode='edge')])
        list_depth.extend([iaa.Affine(10.0 / val_scala, mode='edge')])

        val_rotation = random.randrange(-180, 181, 90)
        list.extend([iaa.Affine(rotate=val_rotation, mode='edge')])
        list_depth.extend([iaa.Affine(rotate=val_rotation, mode='edge')])

        augseq = iaa.Sequential(list)
        batch[i] = augseq.augment_image(batch[i])
        augseq_depth = iaa.Sequential(list)
        batch_depth[i] = augseq_depth.augment_image(batch_depth[i])

        list = []
        list_depth = []


""" Loop for gridsearch of hyper-parameters """

set_params = [lr + nn + bs + aa + mn + do + dt for lr in learning_rate for nn in num_neurons for bs in batch_size for aa
              in l2_factor for mn in maximum_norm for do in dropout_rate for dt in depth_transf]

for hp in set_params:

    lr = hp[0]
    nn = hp[1]
    bs = hp[2]
    aa = hp[3]
    mn = hp[4]
    do = hp[5]
    dt = hp[6]

    """ Data management """

    # Place data loading and preprocessing on the cpu
    # with tf.device('/cpu:0'):
    dataset_train_dir = dataset_train_dir_rgb
    dataset_val_dir = dataset_val_dir_rgb

    tr_data = ImageDataHandler(
        train_file,
        data_dir=dataset_train_dir,
        params_dir=params_root_dir,
        img_size=img_size,
        batch_size=bs,
        num_classes=num_classes,
        shuffle=True,
        random_crops=False)

    val_data = ImageDataHandler(
        val_file,
        data_dir=dataset_val_dir,
        params_dir=params_root_dir,
        img_size=img_size,
        batch_size=bs,
        num_classes=num_classes,
        shuffle=False,
        random_crops=False)

    test_data = ImageDataHandler(
        test_file,
        data_dir=dataset_val_dir,
        params_dir=params_root_dir,
        img_size=img_size,
        batch_size=bs,
        num_classes=num_classes,
        shuffle=False,
        random_crops=False)

    # Create a re-initializable iterator given the dataset structure
    # no need for two different to deal with training and val data,
    # just two initializers
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)
    testing_init_op = iterator.make_initializer(test_data.data)

    # Get the number of training/validation steps per epoch
    tr_batches_per_epoch = int(np.floor(tr_data.data_size / bs))
    val_batches_per_epoch = int(np.floor(val_data.data_size / bs))
    test_batches = int(np.floor(test_data.data_size / bs))

    init_op_rgb = {'training': training_init_op, 'validation': validation_init_op, 'testing': testing_init_op}
    batches_per_epoch = {'training': tr_batches_per_epoch, 'validation': val_batches_per_epoch, 'testing': test_batches}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    # Start Tensorflow session
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count={'GPU':1})) as sess:
    with tf.Session(config=config) as sess:
        ## RGB branch
        # TF placeholder for graph input and output

        x_rgb = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        x_depth = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 3])
        y = tf.placeholder(tf.float32, [None, num_classes])

        x_rgb = tf.reshape(x_rgb, [-1, img_size[0], img_size[1], 3])
        x_depth = tf.reshape(x_depth, [-1, img_size[0], img_size[1], 3])
        y = tf.reshape(y, [-1, num_classes])

        keep_prob = tf.placeholder(tf.float32)
        training_phase = tf.placeholder(tf.bool)

        loss_val = []
        loss_train = []
        acc_val = []
        acc_train = []

        # Max norm
        ckn = np.inf
        rkc = mn  # np.inf
        rrc = mn  # np.inf
        dkc = mn  # np.inf
        conv_kernel_constraint = tf.keras.constraints.MaxNorm(ckn, axis=[0, 1, 2])
        rnn_kernel_constraint = tf.keras.constraints.MaxNorm(rkc, axis=[0, 1])
        rnn_recurrent_constraint = tf.keras.constraints.MaxNorm(rrc, axis=1)
        dense_kernel_constraint = tf.keras.constraints.MaxNorm(dkc, axis=0)

        # Initialize models
        with tf.variable_scope('rgb', reuse=None):
            model_rgb = ResNet(x_rgb, num_classes, mode='rgb')
        with tf.variable_scope('depth', reuse=None):
            model_depth = ResNet(x_depth, num_classes, mode='depth')

        # Extract features
        res1_rgb = model_rgb.relu1
        inter_res2_rgb = model_rgb.inter_res2
        res2_rgb = model_rgb.res2
        inter_res3_rgb = model_rgb.inter_res3
        res3_rgb = model_rgb.res3
        inter_res4_rgb = model_rgb.inter_res4
        res4_rgb = model_rgb.res4
        inter_res5_rgb = model_rgb.inter_res5
        res5_rgb = model_rgb.res5

        pool2_flat_rgb = model_rgb.pool2_flat

        res1_depth = model_depth.relu1
        inter_res2_depth = model_depth.inter_res2
        res2_depth = model_depth.res2
        inter_res3_depth = model_depth.inter_res3
        res3_depth = model_depth.res3
        inter_res4_depth = model_depth.inter_res4
        res4_depth = model_depth.res4
        inter_res5_depth = model_depth.inter_res5
        res5_depth = model_depth.res5

        pool2_flat_depth = model_depth.pool2_flat

        # Conv1x1
        with tf.variable_scope('conv1x1'):

            # depth_transf = 64

            relu_conv1x1_res1_rgb = transformation_block(res1_rgb, dt, conv_kernel_constraint, training_phase,
                                                         'redux_rgb_res1')
            relu_conv1x1_inter_res2_rgb = transf_block(inter_res2_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res2')
            relu_conv1x1_res2_rgb = transf_block(res2_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res2')
            relu_conv1x1_inter_res3_rgb = transf_block(inter_res3_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res3')
            relu_conv1x1_res3_rgb = transf_block(res3_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res3')
            relu_conv1x1_inter_res4_rgb = transf_block(inter_res4_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res4')
            relu_conv1x1_res4_rgb = transf_block(res4_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res4')
            relu_conv1x1_inter_res5_rgb = transf_block(inter_res5_rgb, dt, conv_kernel_constraint, training_phase,
                                                       'redux_rgb_inter_res5')
            relu_conv1x1_res5_rgb = transf_block(res5_rgb, dt, conv_kernel_constraint, training_phase, 'redux_rgb_res5')

            relu_conv1x1_res1_depth = transformation_block(res1_depth, dt, conv_kernel_constraint, training_phase,
                                                           'redux_depth_res1')
            relu_conv1x1_inter_res2_depth = transf_block(inter_res2_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res2')
            relu_conv1x1_res2_depth = transf_block(res2_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res2')
            relu_conv1x1_inter_res3_depth = transf_block(inter_res3_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res3')
            relu_conv1x1_res3_depth = transf_block(res3_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res3')
            relu_conv1x1_inter_res4_depth = transf_block(inter_res4_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res4')
            relu_conv1x1_res4_depth = transf_block(res4_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res4')
            relu_conv1x1_inter_res5_depth = transf_block(inter_res5_depth, dt, conv_kernel_constraint, training_phase,
                                                         'redux_depth_inter_res5')
            relu_conv1x1_res5_depth = transf_block(res5_depth, dt, conv_kernel_constraint, training_phase,
                                                   'redux_depth_res5')

        relu_conv1x1_res1_rgb = tf.reshape(relu_conv1x1_res1_rgb, [-1, flat_shape(relu_conv1x1_res1_rgb)])
        relu_conv1x1_inter_res2_rgb = tf.reshape(relu_conv1x1_inter_res2_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res2_rgb)])
        relu_conv1x1_res2_rgb = tf.reshape(relu_conv1x1_res2_rgb, [-1, flat_shape(relu_conv1x1_res2_rgb)])
        relu_conv1x1_inter_res3_rgb = tf.reshape(relu_conv1x1_inter_res3_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res3_rgb)])
        relu_conv1x1_res3_rgb = tf.reshape(relu_conv1x1_res3_rgb, [-1, flat_shape(relu_conv1x1_res3_rgb)])
        relu_conv1x1_inter_res4_rgb = tf.reshape(relu_conv1x1_inter_res4_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res4_rgb)])
        relu_conv1x1_res4_rgb = tf.reshape(relu_conv1x1_res4_rgb, [-1, flat_shape(relu_conv1x1_res4_rgb)])
        relu_conv1x1_inter_res5_rgb = tf.reshape(relu_conv1x1_inter_res5_rgb,
                                                 [-1, flat_shape(relu_conv1x1_inter_res5_rgb)])
        relu_conv1x1_res5_rgb = tf.reshape(relu_conv1x1_res5_rgb, [-1, flat_shape(relu_conv1x1_res5_rgb)])

        relu_conv1x1_res1_depth = tf.reshape(relu_conv1x1_res1_depth, [-1, flat_shape(relu_conv1x1_res1_depth)])
        relu_conv1x1_inter_res2_depth = tf.reshape(relu_conv1x1_inter_res2_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res2_depth)])
        relu_conv1x1_res2_depth = tf.reshape(relu_conv1x1_res2_depth, [-1, flat_shape(relu_conv1x1_res2_depth)])
        relu_conv1x1_inter_res3_depth = tf.reshape(relu_conv1x1_inter_res3_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res3_depth)])
        relu_conv1x1_res3_depth = tf.reshape(relu_conv1x1_res3_depth, [-1, flat_shape(relu_conv1x1_res3_depth)])
        relu_conv1x1_inter_res4_depth = tf.reshape(relu_conv1x1_inter_res4_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res4_depth)])
        relu_conv1x1_res4_depth = tf.reshape(relu_conv1x1_res4_depth, [-1, flat_shape(relu_conv1x1_res4_depth)])
        relu_conv1x1_inter_res5_depth = tf.reshape(relu_conv1x1_inter_res5_depth,
                                                   [-1, flat_shape(relu_conv1x1_inter_res5_depth)])
        relu_conv1x1_res5_depth = tf.reshape(relu_conv1x1_res5_depth, [-1, flat_shape(relu_conv1x1_res5_depth)])

        # RGB and depth pipelines' merge point
        relu_conv1x1_res1 = tf.concat([relu_conv1x1_res1_rgb, relu_conv1x1_res1_depth], axis=1)
        relu_conv1x1_inter_res2 = tf.concat([relu_conv1x1_inter_res2_rgb, relu_conv1x1_inter_res2_depth], axis=1)
        relu_conv1x1_res2 = tf.concat([relu_conv1x1_res2_rgb, relu_conv1x1_res2_depth], axis=1)
        relu_conv1x1_inter_res3 = tf.concat([relu_conv1x1_inter_res3_rgb, relu_conv1x1_inter_res3_depth], axis=1)
        relu_conv1x1_res3 = tf.concat([relu_conv1x1_res3_rgb, relu_conv1x1_res3_depth], axis=1)
        relu_conv1x1_inter_res4 = tf.concat([relu_conv1x1_inter_res4_rgb, relu_conv1x1_inter_res4_depth], axis=1)
        relu_conv1x1_res4 = tf.concat([relu_conv1x1_res4_rgb, relu_conv1x1_res4_depth], axis=1)
        relu_conv1x1_inter_res5 = tf.concat([relu_conv1x1_inter_res5_rgb, relu_conv1x1_inter_res5_depth], axis=1)
        relu_conv1x1_res5 = tf.concat([relu_conv1x1_res5_rgb, relu_conv1x1_res5_depth], axis=1)

        rnn_input = tf.stack(
            [relu_conv1x1_res1, relu_conv1x1_inter_res2, relu_conv1x1_res2, relu_conv1x1_inter_res3, relu_conv1x1_res3,
             relu_conv1x1_inter_res4, relu_conv1x1_res4, relu_conv1x1_inter_res5, relu_conv1x1_res5], axis=1)

        # Recurrent net
        with tf.variable_scope("rnn"):
            rnn_h = GRU(nn, activation='tanh', dropout=do, recurrent_dropout=do, name="rnn_h",
                        kernel_constraint=rnn_kernel_constraint, recurrent_constraint=rnn_recurrent_constraint)(
                rnn_input)
            preds = Dense(num_classes, activation='softmax', kernel_constraint=dense_kernel_constraint)(rnn_h)

        # Include keras-related metadata in the session
        K.set_session(sess)

        # Define trainable variables
        trainable_variables_rnn = [v for v in tf.trainable_variables(scope="rnn")]
        trainable_variables_conv1x1 = [v for v in tf.trainable_variables(scope="conv1x1")]
        trainable_variables_rgb = [v for v in tf.trainable_variables(scope="rgb")]
        trainable_variables_depth = [v for v in tf.trainable_variables(scope="depth")]

        # Define the training and validation opsi
        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        lr_boundaries = [int(num_epochs * tr_batches_per_epoch * 0.5)]  # , int(num_epochs*tr_batches_per_epoch*0.6)]
        lr_values = [lr, lr / 10]
        decayed_lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)

        # L2-regularization
        alpha_rnn = aa
        alpha_conv1x1 = aa
        # alpha_rgb = aa
        # alpha_depth = aa
        l2_rnn = tf.add_n([tf.nn.l2_loss(tv_rnn) for tv_rnn in trainable_variables_rnn
                           if 'bias' not in tv_rnn.name]) * alpha_rnn
        l2_conv1x1 = tf.add_n([tf.nn.l2_loss(tv_conv1x1) for tv_conv1x1 in trainable_variables_conv1x1
                               if 'bias' not in tv_conv1x1.name]) * alpha_conv1x1
        # l2_rgb = tf.add_n([tf.nn.l2_loss(tv_rgb) for tv_rgb in trainable_variables_rgb
        #                      if 'bias' not in tv_rgb.name]) * alpha_rgb
        # l2_depth = tf.add_n([tf.nn.l2_loss(tv_depth) for tv_depth in trainable_variables_depth
        #                      if 'bias' not in tv_depth.name]) * alpha_depth

        # F2-norm
        # rgb_nodes = [relu_conv1x1_inter_res2_rgb, relu_conv1x1_res2_rgb, relu_conv1x1_inter_res3_rgb, relu_conv1x1_res3_rgb, relu_conv1x1_inter_res4_rgb, relu_conv1x1_res4_rgb, relu_conv1x1_inter_res5_rgb, relu_conv1x1_res5_rgb]
        rgb_nodes = [relu_conv1x1_res1_rgb, relu_conv1x1_inter_res2_rgb, relu_conv1x1_res2_rgb,
                     relu_conv1x1_inter_res3_rgb, relu_conv1x1_res3_rgb, relu_conv1x1_inter_res4_rgb,
                     relu_conv1x1_res4_rgb, relu_conv1x1_inter_res5_rgb, relu_conv1x1_res5_rgb]
        # depth_nodes = [relu_conv1x1_inter_res2_depth, relu_conv1x1_res2_depth, relu_conv1x1_inter_res3_depth, relu_conv1x1_res3_depth, relu_conv1x1_inter_res4_depth, relu_conv1x1_res4_depth, relu_conv1x1_inter_res5_depth, relu_conv1x1_res5_depth]
        depth_nodes = [relu_conv1x1_res1_depth, relu_conv1x1_inter_res2_depth, relu_conv1x1_res2_depth,
                       relu_conv1x1_inter_res3_depth, relu_conv1x1_res3_depth, relu_conv1x1_inter_res4_depth,
                       relu_conv1x1_res4_depth, relu_conv1x1_inter_res5_depth, relu_conv1x1_res5_depth]
        reg = tf.Variable(0.0, trainable=False)

        # sigm_reg = 0.0001*tf.sigmoid(((12*tf.cast(global_step,tf.float32))/(num_epochs*tr_batches_per_epoch))-6, name='reg_sigmoid')
        sigm_reg = 1.0
        reg_values = [sigm_reg, 0.0]
        decayed_reg = tf.train.piecewise_constant(global_step, lr_boundaries, reg_values)
        recompute_reg = tf.assign(reg, sigm_reg)

        reg_lambda = [0.0 * decayed_reg, 0.0 * decayed_reg, 1e-6 * decayed_reg, 1e-5 * decayed_reg, 1e-5 * decayed_reg,
                      1e-4 * decayed_reg, 1e-4 * decayed_reg, 0.0 * decayed_reg, 0.0 * decayed_reg]

        lr_mult_conv1x1 = decayed_reg

        # Loss
        loss_l2 = l2_rnn + l2_conv1x1  # + l2_rgb + l2_depth
        loss_cls = tf.reduce_mean(categorical_crossentropy(y, preds))

        loss = loss_cls  # + loss_l2
        train_step_rnn = tf.keras.optimizers.RMSprop(lr=decayed_lr).get_updates(loss=loss,
                                                                                params=trainable_variables_rnn)
        train_step_conv1x1 = tf.keras.optimizers.RMSprop(lr=decayed_lr * lr_mult_conv1x1).get_updates(loss=loss,
                                                                                                      params=trainable_variables_conv1x1)

        # train_step_rgb = tf.keras.optimizers.RMSprop(lr=decayed_lr).get_updates(loss=loss,params=trainable_variables_rgb)

        # train_step_depth = tf.keras.optimizers.RMSprop(lr=decayed_lr).get_updates(loss=loss,params=trainable_variables_depth)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.group(train_step_rnn, train_step_conv1x1, increment_global_step)
            # train_step = tf.group(train_step_rnn, train_step_conv1x1, train_step_rgb, train_step_depth, increment_global_step, recompute_reg)

        accuracy = tf.reduce_mean(categorical_accuracy(y, preds))

        # Create summaries for Tensorboard
        tf.summary.scalar("loss_cls", loss_cls)
        tf.summary.scalar("loss_l2", loss_l2)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("learning rate", decayed_lr)
        tf.summary.scalar("lambda", decayed_reg)
        summary_op = tf.summary.merge_all()

        name = str(lr) + '_' + str(bs) + '_' + str(nn)
        train_writer = tf.summary.FileWriter(tensorboard_log + name + '/train/', graph=sess.graph)
        val_writer = tf.summary.FileWriter(tensorboard_log + name + '/val/')

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        best_saver = tf.train.Saver(max_to_keep=1)

        # Load the pretrained weights into the non-trainable layer
        model_rgb.load_params(sess, params_dir_rgb, trainable=True)
        model_depth.load_params(sess, params_dir_depth, trainable=True)

        print("\nHyper-parameters: lr={}, #neurons={}, bs={}, l2={}, max_norm={}, dropout_rate={}, num_class={}".format(
                lr, nn, bs, aa,mn, do, num_classes))
        print("Number of trainable parameters = {}".format(
            count_params(trainable_variables_rnn) + count_params(trainable_variables_conv1x1) +
            count_params(trainable_variables_rgb) + count_params(trainable_variables_depth)))

        print("\n{} Generate features from training set".format(datetime.now()))
        
        makelog = "\nHyper-parameters: lr={}, #neurons={}, bs={}, l2={}, max_norm={}, dropout_rate={}, num_class={}".format(
                lr, nn, bs, aa,mn, do, num_classes) + "\nNumber of trainable parameters = {}".format(
            count_params(trainable_variables_rnn) + count_params(trainable_variables_conv1x1) +
            count_params(trainable_variables_rgb) + count_params(trainable_variables_depth)) + \
                  "\n{} Generate features from training set".format(datetime.now())
        sLog(makelog,simpleLog)

        tb_train_count = 0
        tb_val_count = 0
        best_loss_val = 1000

        # Loop over number of epochs
        num_samples = 0
        # Training set

        sess.run(training_init_op)

        # Progress bar setting
        bar = progressbar.ProgressBar(maxval=tr_batches_per_epoch,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        train_loss = 0
        train_acc = 0
        for i in range(tr_batches_per_epoch):
            bar.update(i + 1)
            tb_train_count += 1
            rgb_batch, depth_batch, label_batch = sess.run(next_batch)

            num_samples += np.shape(rgb_batch)[0]
            feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: (1 - do),
                         training_phase: True, K.learning_phase(): 1}
            batch_loss, summary = sess.run([loss, summary_op], feed_dict=feed_dict)
            train_loss += batch_loss
            train_writer.add_summary(summary, tb_train_count)

        bar.finish()

        train_loss /= tr_batches_per_epoch  # num_samples
        print("training loss = {}\n".format(train_loss))
        makelog = "training loss = {}\n".format(train_loss)
        sLog(str(makelog),simpleLog)

        val_acc = 0
        val_loss = 0
        num_samples = 0

        sess.run(validation_init_op)
        for i in range(val_batches_per_epoch):
            tb_val_count += 1
            rgb_batch, depth_batch, label_batch = sess.run(next_batch)
            num_samples += np.shape(rgb_batch)[0]

            feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: 1.0, training_phase: False,
                         K.learning_phase(): 0}
            batch_loss, batch_acc, summary = sess.run([loss, accuracy, summary_op], feed_dict=feed_dict)

            val_loss += batch_loss
            val_acc += batch_acc
            val_writer.add_summary(summary, tb_val_count * (tr_batches_per_epoch / val_batches_per_epoch))

        val_loss /= val_batches_per_epoch  # num_samples
        val_acc /= val_batches_per_epoch  # num_samples

        print("\n{} Validation loss : {}, Validation Accuracy = {:.4f}".format(datetime.now(), val_loss, val_acc))
        makelog = "\n{} Validation loss : {}, Validation Accuracy = {:.4f}".format(datetime.now(), val_loss, val_acc)
        sLog(str(makelog), simpleLog)

        # Loop over number of epochs
        for epoch in range(num_epochs):
            num_samples = 0

            # Training set
            print("\nEpoch: {}/{}".format(epoch + 1, num_epochs))
            makelog = "\nEpoch: {}/{}".format(epoch + 1, num_epochs)
            sLog(str(makelog), simpleLog)
            
            sess.run(training_init_op)

            # Progress bar setting
            bar = progressbar.ProgressBar(maxval=tr_batches_per_epoch,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            train_loss = 0
            train_acc = 0

            for i in range(tr_batches_per_epoch):
                bar.update(i + 1)
                tb_train_count += 1
                rgb_batch, depth_batch, label_batch = sess.run(next_batch)

                num_samples += np.shape(rgb_batch)[0]

                # apply data augmentation
                data_aug(rgb_batch, depth_batch)

                feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: (1 - do),
                             training_phase: True, K.learning_phase(): 1}
                batch_loss, _, summary, batch_acc_train = sess.run([loss, train_step, summary_op, accuracy],
                                                                   feed_dict=feed_dict)
                train_loss += batch_loss
                train_acc += batch_acc_train

                train_writer.add_summary(summary, tb_train_count)

            bar.finish()

            train_loss /= tr_batches_per_epoch  # num_samples
            train_acc /= tr_batches_per_epoch
            loss_train.append(train_loss)
            acc_train.append(train_acc)

            print("training loss = {}\n".format(train_loss))
            makelog = "training loss = {}\n".format(train_loss)
            sLog(str(makelog), simpleLog)

            if (epoch + 1) % 1 == 0:
                sess.run(validation_init_op)
                num_samples = 0
                val_loss = 0
                val_acc = 0

                for i in range(val_batches_per_epoch):
                    tb_val_count += 1
                    rgb_batch, depth_batch, label_batch = sess.run(next_batch)

                    num_samples += np.shape(rgb_batch)[0]

                    feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: 1.0,
                                 training_phase: False, K.learning_phase(): 0}
                    batch_loss, batch_acc, summary, batch_preds = sess.run([loss, accuracy, summary_op, preds],
                                                                           feed_dict=feed_dict)

                    val_loss += batch_loss
                    val_acc += batch_acc
                    val_writer.add_summary(summary, tb_val_count * (tr_batches_per_epoch / val_batches_per_epoch))

                val_loss /= val_batches_per_epoch  # num_samples
                val_acc /= val_batches_per_epoch  # num_samples
                loss_val.append(val_loss)
                acc_val.append(val_acc)

                print("\n{} Validation loss : {}, Validation Accuracy = {:.4f}".format(datetime.now(), val_loss, val_acc))
                
                makelog = "\n{} Validation loss : {}, Validation Accuracy = {:.4f}".format(datetime.now(), val_loss, val_acc)
                sLog(str(makelog), simpleLog)
                
                if val_loss < best_loss_val:
                    best_saver.save(sess, "/content/drive/My Drive/RCFusionGPU/bestmodel/model")
                    best_loss_val = val_loss
                    print("epoca migliore:", epoch +1)
            
                # Early stopping
                if epoch > patience:
                    count = 0
                    for x in loss_val[epoch -patience: epoch]:
                        if val_loss > x - delta:
                            count +=1
                    if count == patience:
                        num_epochs = epoch + 1
                        break
                            

            # Early stopping for ill-posed params combination
            if ((epoch == 0) and (val_acc < 0.2)) or ((epoch == 9) and (val_acc < 0.5)) or np.isnan(train_loss):
                print("Training stopped due to poor results or divergence: validation accuracy = {}".format(val_acc))
                break
            
                makelog = "Training stopped due to poor results or divergence: validation accuracy = {}".format(val_acc)
                sLog(str(makelog), simpleLog)
            
            
        fig1 = plt.figure(1)
        plt.title('Loss')
        plt.plot(loss_val, 'r', label='Validation Loss')
        plt.plot(loss_train, 'b', label='Training Loss')
        plt.legend(loc="upper right")
        x = list(range(0, num_epochs+1, 5))
        plt.xlim(right=num_epochs)
        plt.xticks(x)
        plt.grid(True)
        fig1.savefig("loss.png")
        plt.show()


        fig2 = plt.figure(2)
        plt.title('Accuracy')
        plt.plot(acc_val, 'r', label='Validation Accuracy')
        plt.plot(acc_train, 'b', label='Training Accuracy')
        plt.legend(loc="lower right")
        x = list(range(0, num_epochs+1, 5))
        plt.xlim(right=num_epochs)
        plt.xticks(x)
        plt.grid(True)
        fig2.savefig("accuracy.png")
        plt.show()
        
        #TEST PHASE
        #tf.reset_default_graph()
        #saver = tf.train.import_meta_graph(sess, "/content/drive/My Drive/RCFusionGPU/bestmodel/model.meta")
        saver = tf.train.Saver()
        saver.restore(sess, "/content/drive/My Drive/RCFusionGPU/bestmodel/model")
        print("Model restored")
        sess.run(testing_init_op)
        num_samples = 0
        test_loss = 0
        test_acc = 0
        test_val_count = 0
        full_batch = []
        full_pred = []
        for i in range(test_batches):
            test_val_count += 1
            rgb_batch, depth_batch, label_batch = sess.run(next_batch)

            num_samples += np.shape(rgb_batch)[0]

            feed_dict = {x_rgb: rgb_batch, x_depth: depth_batch, y: label_batch, keep_prob: 1.0, training_phase: False,
                         K.learning_phase(): 0}
            batch_loss, batch_acc, summary, batch_preds = sess.run([loss, accuracy, summary_op, preds],
                                                                   feed_dict=feed_dict)

            test_loss += batch_loss
            test_acc += batch_acc
            full_pred.extend(batch_preds)
            full_batch.extend(label_batch)
            
        test_loss /= test_batches# num_samples
        test_acc /= test_batches

        #print(full_batch)
        batchMatrix = defineImageRank(full_pred, full_batch)
        classList, num_of_frames = countTry(batchMatrix, num_classes)
        values = cmc_values(classList, num_of_frames)
        plot_cmc(values, "CMC", num_classes)
        computeRoc(full_batch,full_pred,num_classes)

        print("\n{} Testing loss : {}, Testing Accuracy = {:.4f}".format(datetime.now(), test_loss, test_acc))
        
        makelog = "\n{} Testing loss : {}, Testing Accuracy = {:.4f}".format(datetime.now(), test_loss, test_acc)
        sLog(str(makelog), simpleLog)

    tf.reset_default_graph()
    # shutil.rmtree(tensorboard_log)
    # tensorboard --logdir=/tmp/tensorflow/