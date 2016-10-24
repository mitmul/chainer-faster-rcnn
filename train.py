#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import training
from utils.prepare_train import create_args
from utils.prepare_train import create_logger
from utils.prepare_train import create_result_dir
from utils.prepare_train import get_model
from utils.prepare_train import get_optimizer

import chainer

if __name__ == '__main__':
    args = create_args()
    result_dir = create_result_dir(args.model_name)
    create_logger(args, result_dir)
    get_model(args.model_file, args.model_name, n_classes, train=False, trunk_path=None,
                  result_dir=None
