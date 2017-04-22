#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Shunta Saito

import matplotlib  # isort:skip

matplotlib.use('Agg')  # isort:skip
import sys  # isort:skip

sys.path.insert(0, '.')  # isort:skip

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.dataset import concat_examples
from chainer.training import extensions
from datasets.pascal_voc_dataset import VOC
from models.faster_rcnn import FasterRCNN


def warmup(model, iterator, gpu_id=0):
    batch = iterator.next()
    img, img_info, bbox = concat_examples(batch, gpu_id)
    img = chainer.Variable(img)
    img_info = chainer.Variable(img_info)
    bbox = chainer.Variable(bbox)
    model.rcnn_train = True
    model(img, img_info, bbox)
    model.rpn_train = True
    model(img, img_info, bbox)


if __name__ == '__main__':
    batchsize = 1

    train_dataset = VOC('train')
    valid_dataset = VOC('val')

    train_iter = iterators.SerialIterator(train_dataset, batchsize)
    model = FasterRCNN()

    chainer.serializers.load_npz('train_rpn/snapshot_571000', model)

    model.to_gpu(0)

    warmup(model, train_iter)
    model.rcnn_train = True

    # optimizer = optimizers.Adam()
    # optimizer.setup(model)
    optimizer = optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (100, 'epoch'), out='train_rcnn')
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/loss_cls',
        'main/cls_accuracy',
        'main/loss_bbox',
        'main/loss_rcnn',
        'elapsed_time',
    ]), trigger=(100, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'snapshot_{.updater.iteration}'),
        trigger=(1000, 'iteration'))
    trainer.extend(extensions.PlotReport(['main/loss_rcnn'],
                                         trigger=(100, 'iteration')))
    trainer.extend(extensions.PlotReport(['main/cls_accuracy'],
                                         trigger=(100, 'iteration')))
    trainer.extend(
        extensions.dump_graph('main/loss_rcnn', out_name='loss_rcnn.dot'))

    trainer.run()
