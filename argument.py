import argparse

from chainer import cuda
import numpy

def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a Caffe reference model on ILSVRC2012 dataset')
    parser.add_argument('model_type', choices=('alexnet', 'caffenet', 'googlenet', 'nin', 'vgg'),
                        help='Model type (alexnet, caffenet, googlenet, nin, vgg)')
    parser.add_argument('model', help='Path to the pretrained Caffe model')
    parser.add_argument('--src', '-s', default='./src',
                        help='source path for images')
    parser.add_argument('--dst', '-d', default='./dst',
                        help='distination path for features')
    parser.add_argument('--mean', '-m', default='~/caffemodels/ilsvrc_2012_mean.npy',
                        help='Path to the mean file')
    parser.add_argument('--batchsize', '-B', type=int, default=1024,
                        help='Minibatch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='Zero-origin GPU ID (nevative value indicates CPU)')    

    args = parser.parse_args()
    assert(args.batchsize > 0)        

    xp = numpy
    if args.gpu >= 0:
        cuda.check_cuda_available()
        xp = cuda.cupy

    def slash(d):
        return d if d[-1] is '/' else d + '/'

    args.src = slash(args.src)
    args.dst = slash(args.dst)
    return args, xp
