#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Shunta Saito

import argparse
import json

import matplotlib  # isort:skip

matplotlib.use('Agg')  # isort:skip

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from datasets.pascal_voc_dataset import VOC
from models.faster_rcnn import FasterRCNN


def warmup(model, gpu_ids):
    train_dataset = VOC('train')
    iterator = iterators.MultiprocessIterator(train_dataset, 1,
                                              shared_mem=10000000)

    batch = iterator.next()
    img, img_info, bbox = batch[0]
    img = chainer.Variable(img[None, ...])
    img_info = chainer.Variable(img_info[None, ...])
    bbox = chainer.Variable(bbox[None, ...])
    for gpu_id in gpu_ids:
        if gpu_id >= 0:
            img.to_gpu(gpu_id)
            img_info.to_gpu(gpu_id)
            bbox.to_gpu(gpu_id)
            model.to_gpu(gpu_id)
        model.rcnn_train = True
        model(img, img_info, bbox)
        model.rpn_train = True
        model(img, img_info, bbox)
        if gpu_id >= 0:
            model.to_cpu()


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--mode', type=str, default='rpn',
                        choices=['rpn', 'rcnn'])
    parser.add_argument('--gpus', nargs='*', type=int, default=0)
    parser.add_argument('--snapshot_iter', type=int, default=10000)
    parser.add_argument('--lr_drop_iter', type=int, default=60000)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--stop_iter', type=int, default=180000)
    parser.add_argument('--report_iter', type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(vars(args), sort_keys=True, indent=4))
    return args


def create_lrdrop_ext(gamma):
    @training.make_extension()
    def learning_rate_drop(trainer):
        trainer.updater.get_optimizer('main').lr *= gamma

    return learning_rate_drop


def train_mode(updater, mode, lr_drop_iter, snapshot_iter, report_iter,
               stop_iter):
    trainer = training.Trainer(updater, (stop_iter, 'iteration'),
                               out='results')
    trainer.extend(
        extensions.LogReport(trigger=(report_iter, 'iteration')))
    trainer.extend(extensions.observe_lr(),
                   trigger=(report_iter, 'iteration'))
    trainer.extend(create_lrdrop_ext(args.gamma),
                   trigger=(lr_drop_iter, 'iteration'))
    if mode == 'rpn':
        updater.get_optimizer('main').target.rpn_train = True
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration',
            'main/RPN/rpn_loss',
            'main/RPN/rpn_loss_cls',
            'main/RPN/rpn_cls_accuracy',
            'main/RPN/rpn_loss_bbox',
            'elapsed_time',
            'lr',
        ]), trigger=(report_iter, 'iteration'))
        trainer.extend(extensions.ProgressBar(),
                       trigger=(report_iter, 'iteration'))
        trainer.extend(extensions.PlotReport(
            ['main/RPN/rpn_loss'],
            trigger=(report_iter, 'iteration')))
        trainer.extend(
            extensions.dump_graph('main/RPN/rpn_loss',
                                  out_name='rpn_loss.dot'))

        # Add snapshot extensions
        trainer.extend(
            extensions.snapshot(
                filename='rpn_trainer_snapshot_{.updater.iteration}'),
            trigger=(snapshot_iter, 'iteration'))
        trainer.extend(
            extensions.snapshot_object(
                model, 'rpn_model_snapshot_{.updater.iteration}'),
            trigger=(snapshot_iter, 'iteration'))
    elif mode == 'rcnn':
        updater.get_optimizer('main').target.rcnn_train = True
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration',
            'main/loss_cls',
            'main/cls_accuracy',
            'main/loss_bbox',
            'main/loss_rcnn',
            'elapsed_time',
            'lr',
        ]), trigger=(report_iter, 'iteration'))
        trainer.extend(extensions.ProgressBar(),
                       trigger=(report_iter, 'iteration'))
        trainer.extend(extensions.PlotReport(
            ['main/RPN/rpn_loss'],
            trigger=(report_iter, 'iteration')))
        trainer.extend(
            extensions.dump_graph('main/RPN/rpn_loss',
                                  out_name='rpn_loss.dot'))

        # Add snapshot extensions
        trainer.extend(
            extensions.snapshot(
                filename='rpn_trainer_snapshot_{.updater.iteration}'),
            trigger=(snapshot_iter, 'iteration'))
        trainer.extend(
            extensions.snapshot_object(
                model, 'rpn_model_snapshot_{.updater.iteration}'),
            trigger=(snapshot_iter, 'iteration'))

    trainer.run()
    del trainer


if __name__ == '__main__':
    args = create_args()
    chainer.cuda.get_device_from_id(args.gpus[0]).use()

    model = FasterRCNN()
    devices = {'main': args.gpus[0]}
    if len(args.gpus) > 1:
        devices.update(dict(('gpu{}'.format(i), i) for i in args.gpus[1:]))

    warmup(model, devices.values())
    if args.mode == 'rpn':
        model.rpn_train = True
    elif args.mode == 'rcnn':
        model.rcnn_train = True

    train_dataset = VOC('train')
    valid_dataset = VOC('val')

    train_iter = iterators.MultiprocessIterator(
        train_dataset, len(devices), shared_mem=10000000)

    optimizer = optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    if len(devices) == 1:
        updater = training.StandardUpdater(train_iter, optimizer,
                                           device=devices['main'])
    else:
        updater = training.ParallelUpdater(train_iter, optimizer,
                                           devices=devices)

    # updater, mode, lr_drop_iter, snapshot_iter, report_iter,
    # stop_iter
    train_mode(updater, 'rpn', 10, 10, 10, 20)
    train_mode(updater, 'rcnn', 10, 10, 10, 40)
