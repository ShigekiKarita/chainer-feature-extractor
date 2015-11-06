from __future__ import print_function
import sys

from chainer import cuda
from chainer.functions import caffe
import numpy


def load_model(args):
    print('Loading Caffe model file %s...' % args.model, file=sys.stderr)
    func = caffe.CaffeFunction(args.model)
    print('Loaded', file=sys.stderr)
    print("Network.layers:", func.layers)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        func.to_gpu()

    in_size = 0
    mean_image = None
    fun = None
    if args.model_type == 'alexnet' or args.model_type == 'caffenet':
        in_size = 227
        mean_image = numpy.load(args.mean)
        fun = lambda x: func(
            inputs={'data': x}, 
            outputs=['fc8'],
            train=False)[0]

    elif args.model_type == 'nin':
        in_size = 227
        mean_image = numpy.load(args.mean)
        fun = lambda x: func(
            inputs={'data': x},
            outputs=['pool4'], 
            train=False)[0]        

    elif args.model_type == 'googlenet':
        in_size = 224
        # Constant mean over spatial pixels
        # https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt#L13
        mean_image = numpy.ndarray((3, 256, 256), dtype=numpy.float32)
        mean_image[0] = 104
        mean_image[1] = 117
        mean_image[2] = 123
        fun = lambda x: func(
            inputs={'data': x},
            outputs=['loss3/classifier'],
            disable=['loss1/ave_pool', 'loss2/ave_pool'],
            train=False)[0]

    elif args.model_type == 'vgg':
        in_size = 224
        # Constant mean over spatial pixels
        # https://gist.github.com/ksimonyan/211839e770f7b538e2d8#description
        mean_image = numpy.ndarray((3, 256, 256), dtype=numpy.float32)
        mean_image[0] = 103.939
        mean_image[1] = 116.779
        mean_image[2] = 123.68
        fun = lambda x: func(
            inputs={'data': x},
            outputs=['fc8'],
            train=False)[0]
    
    cropwidth = 256 - in_size
    start = cropwidth // 2
    stop = start + in_size
    mean_image = mean_image[:, start:stop, start:stop].copy()

    return fun, in_size, mean_image
