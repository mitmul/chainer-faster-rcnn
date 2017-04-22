#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

if True:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--x_axis', type=str, default='iteration')
    parser.add_argument('--start_x', type=int, default=0)
    args = parser.parse_args()

    for log_fn in sorted(glob.glob('{}/log*'.format(args.result_dir))):
        logs = json.load(open(log_fn))
        loss = defaultdict(list)
        for log in logs:
            if args.x_axis == 'iteration' and log['iteration'] < args.start_x:
                continue
            elif args.x_axis == 'epoch' and log['epoch'] < args.start_x:
                continue
            for k, v in log.items():
                if 'epoch' in k or 'iteration' in k or 'elapsed_time' in k:
                    continue
                loss[k].append([log[args.x_axis], v])
        for k, v in loss.items():
            y = np.array(sorted(v))

            f = plt.figure()
            a = f.add_subplot(111)
            a.set_xlabel(args.x_axis)
            a.set_ylabel(k)

            x = np.where(y[:, 0] >= args.start_x)[0][0]
            a.plot(y[x:, 0], y[x:, 1], label=k)
            l = a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            f.savefig(
                os.path.join(args.result_dir,
                             '{}.png'.format(k.replace('/', '_'))),
                bbox_extra_artists=(l,), bbox_inches='tight')
