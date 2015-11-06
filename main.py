#!/usr/bin/env python
"""Example code of feature extraction with a Caffe model for ILSVRC2012 task.

usage:
    ./main.py nin ~/caffemodels/nin --src ./src --dst ./dst_nin
    ./main.py alexnet ~/caffemodels/alexnet --src ./src --dst ./dst_alexnet
"""
from __future__ import print_function
import os
import sys

import chainer
import numpy

from argument import get_args
from model import load_model
from files import load_image, save_features, mkdir, grep_images


if __name__ == '__main__':
    args, xp = get_args()
    forward, in_size, mean_image = load_model(args)
    mkdir(args.dst)

    msg = True
    ps = []
    path_list = grep_images(args.src)
    total = len(path_list)
    x_batch = numpy.ndarray((args.batchsize, 3, in_size, in_size), dtype=numpy.float32)

    for i, path in enumerate(path_list):
        image = load_image(path, in_size, mean_image)
        x_batch[i % args.batchsize] = image
        ps.append(path)

        if i == 0 and i != total - 1:
            continue
        if (i + 1) % args.batchsize == 0 or i == total - 1:
            x_data = xp.asarray(x_batch)
            x = chainer.Variable(x_data, volatile=True)
            y = forward(x)

            if msg:
                msg = False
                print("resize-cropped image shape: ", x.data.shape[1:])
                print("extracted feature shape: ", y.data.shape[1:])

            print('{} / {}'.format(i, total), end='\r', file=sys.stderr)
            sys.stderr.flush()

            save_features(args.dst, ps, y)
            del x, y
            ps = []
