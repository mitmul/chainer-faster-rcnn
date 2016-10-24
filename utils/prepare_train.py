#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import optimizers
from chainer import serializers

import argparse
import chainer
import chainer.links as L
import imp
import logging
import numpy as np
import os
import shutil
import sys
import time


def create_args():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument(
        '--model_file', type=str,
        help='The model filename with .py extension')
    parser.add_argument(
        '--model_name', type=str,
        help='The model class name')
    parser.add_argument(
        '--resume_model', type=str,
        help='Saved parameter file to be used for resuming')
    parser.add_argument(
        '--resume_opt', type=str,
        help='Saved optimization file to be used for resuming')
    parser.add_argument(
        '--trunk', type=str,
        help='The pre-trained trunk param filename with .npz extension')
    parser.add_argument(
        '--epoch', type=int, default=100,
        help='When the trianing will finish')
    parser.add_argument(
        '--gpus', type=str, default='0,1,2,3',
        help='GPU Ids to be used')
    parser.add_argument(
        '--batchsize', type=int, default=128,
        help='minibatch size')
    parser.add_argument(
        '--snapshot_iter', type=int, default=None,
        help='The current learnt parameters in the model is saved every'
             'this iteration')
    parser.add_argument(
        '--valid_freq', type=int, default=1,
        help='Perform test every this iteration (0 means no test)')
    parser.add_argument(
        '--valid_batchsize', type=int, default=256,
        help='The mini-batch size during validation loop')
    parser.add_argument(
        '--show_log_iter', type=int, default=10,
        help='Show loss value per this iterations')

    # Settings
    parser.add_argument(
        '--shift_jitter', type=int, default=None,
        help='Shift jitter amount for data augmentation (typically 50)')
    parser.add_argument(
        '--scale_jitter', type=float, default=None,
        help='Scale jitter amount for data augmentation '
             '(typically +- 2 ** 0.25)')
    parser.add_argument(
        '--scale', type=float, default=0.5,
        help='Scale for the input images and label images')
    parser.add_argument(
        '--n_classes', type=int, default=20,
        help='The number of classes that the model predicts')
    parser.add_argument(
        '--mean', type=str, default=None,
        help='Mean npy over the training data')
    parser.add_argument(
        '--std', type=str, default=None,
        help='Stddev npy over the training data')

    # Dataset paths
    parser.add_argument(
        '--train_img_dir', type=str, help='Full path to images for trianing')
    parser.add_argument(
        '--valid_img_dir', type=str, help='Full path to images for validation')
    parser.add_argument(
        '--train_lbl_dir', type=str, help='Full path to labels for trianing')
    parser.add_argument(
        '--valid_lbl_dir', type=str, help='Full path to labels for validation')

    # Optimization settings
    parser.add_argument(
        '--opt', type=str, default='Adam',
        choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop'],
        help='Optimization method')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--adam_alpha', type=float, default=0.001)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument(
        '--lr_decay_freq', type=int, default=10,
        help='The learning rate will be decreased every this epoch')
    parser.add_argument(
        '--lr_decay_ratio', type=float, default=0.1,
        help='When the learning rate is decreased, this number will be'
             'multiplied')
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args()
    xp = chainer.cuda.cupy if chainer.cuda.available else np
    xp.random.seed(args.seed)
    return args


def create_result_dir(model_name):
    result_dir = 'results/{}_{}'.format(
        model_name, time.strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(result_dir):
        result_dir += '_{}'.format(time.clock())
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def create_logger(args, result_dir):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    msg_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(msg_format)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    fileHandler = logging.FileHandler("{}/stdout.log".format(result_dir))
    fileHandler.setFormatter(formatter)
    root.addHandler(fileHandler)
    logging.info(sys.version_info)
    logging.info('chainer version: {}'.format(chainer.__version__))
    logging.info('cuda: {}, cudnn: {}'.format(
        chainer.cuda.available, chainer.cuda.cudnn_enabled))
    logging.info(args)


def get_model(model_file, model_name, train=False, trunk_path=None,
              result_dir=None):
    model = imp.load_source(model_name, model_file)
    model = getattr(model, model_name)

    # Initialize
    model = model(n_classes)
    if trunk_path is not None:
        serializers.load_npz(trunk_path, model)
    if train:
        model = L.Classifier(model)

    # Copy files
    if result_dir is not None:
        base_fn = os.path.basename(model_file)
        dst = '{}/{}'.format(result_dir, base_fn)
        if not os.path.exists(dst):
            shutil.copy(model_file, dst)

    return model


def get_optimizer(model, opt, lr=None, adam_alpha=None, adam_beta1=None,
                  adam_beta2=None, adam_eps=None, weight_decay=None):
    if opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optimizers.Adam(
            alpha=adam_alpha, beta1=adam_beta1,
            beta2=adam_beta2, eps=adam_eps)
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=lr)
    elif opt == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        raise Exception('No optimizer is selected')

    # The first model as the master model
    optimizer.setup(model)
    if opt == 'MomentumSGD':
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(weight_decay))

    return optimizer
