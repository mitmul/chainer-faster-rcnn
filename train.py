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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
