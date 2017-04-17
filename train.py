#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

import matplotlib  # isort:skip

matplotlib.use('Agg')

import json

from chainer import iterators
from chainer import training
from chainer.training import extensions
from datasets.pascal_voc_dataset import VOC
from utils.prepare_train import create_args
from utils.prepare_train import create_result_dir
from utils.prepare_train import get_model
from utils.prepare_train import get_optimizer

if __name__ == '__main__':
    args = create_args()
    result_dir = create_result_dir(args.model_name)
    json.dump(vars(args), open('{}/args.json'.format(result_dir), 'w'))
    print(json.dumps(vars(args), sort_keys=True, indent=4))

    # Prepare devices
    devices = {}
    for gid in [int(i) for i in args.gpus.split(',')]:
        if 'main' not in devices:
            devices['main'] = gid
        else:
            devices['gpu{}'.format(gid)] = gid

    # Instantiate a model
    model = get_model(
        args.model_file, args.model_name, devices['main'], args.rpn_in_ch,
        args.rpn_out_ch, args.n_anchors, args.feat_stride, args.anchor_scales,
        args.num_classes, args.spatial_scale, args.rpn_sigma, args.sigma,
        args.trunk_model, True, result_dir)

    # Instantiate a optimizer
    optimizer = get_optimizer(
        model, args.opt, args.lr, args.adam_alpha, args.adam_beta1,
        args.adam_beta2, args.adam_eps, args.weight_decay)

    # Setting up datasets
    train = VOC('train', False)
    valid = VOC('val', False)
    print('train: {}, valid: {}'.format(len(train), len(valid)))

    # Iterator
    train_iter = iterators.MultiprocessIterator(train, args.batchsize)
    valid_iter = iterators.MultiprocessIterator(
        valid, args.valid_batchsize, repeat=False, shuffle=False)

    # Updater
    updater = ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)

    # Extentions
    trainer.extend(
        extensions.Evaluator(valid_iter, model, device=devices['main']),
        trigger=(args.valid_freq, 'epoch'))
    trainer.extend(extensions.dump_graph(
        'main/rpn_loss_cls', out_name='rpn_loss_cls.dot'))
    trainer.extend(extensions.dump_graph(
        'main/rpn_loss_bbox', out_name='rpn_loss_bbox.dot'))
    trainer.extend(extensions.dump_graph(
        'main/loss_cls', out_name='loss_cls.dot'))
    trainer.extend(extensions.dump_graph(
        'main/loss_bbox', out_name='loss_bbox.dot'))
    trainer.extend(
        extensions.snapshot(trigger=(args.snapshot_iter, 'iteration')))
    trainer.extend(
        extensions.LogReport(trigger=(args.show_log_iter, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/rpn_loss_cls', 'main/rpn_loss_bbox',
         'main/loss_cls', 'main/loss_bbox', 'validation/main/rpn_loss_cls',
         'validation/main/rpn_loss_bbox', 'validation/main/loss_cls',
         'validation/main/loss_bbox']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
