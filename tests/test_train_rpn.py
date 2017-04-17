#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib  # isort:skip

matplotlib.use('Agg')  # isort:skip

import sys  # isort:skip

sys.path.insert(0, '.')  # isort:skip

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from datasets.pascal_voc_dataset import VOC
from models.faster_rcnn import FasterRCNN
from chainer.dataset import concat_examples


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
    model.to_gpu(0)

    warmup(model, train_iter)
    model.rpn_train = True

    # optimizer = optimizers.Adam()
    # optimizer.setup(model)
    optimizer = optimizers.MomentumSGD(lr=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (100, 'epoch'), out='tests/train_test')
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/RPN/rpn_loss',
        'main/RPN/rpn_loss_cls',
        'main/RPN/rpn_cls_accuracy',
        'main/RPN/rpn_loss_bbox',
        'elapsed_time',
    ]), trigger=(100, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'snapshot_{.updater.iteration}'),
        trigger=(1000, 'iteration'))
    trainer.extend(extensions.PlotReport(['main/RPN/rpn_loss'],
                                         trigger=(100, 'iteration')))
    trainer.extend(
        extensions.dump_graph('main/RPN/rpn_loss', out_name='rpn_loss.dot'))

    trainer.run()
