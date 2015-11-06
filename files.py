import fnmatch
import os

from PIL import Image
import numpy
import chainer


def resize_max(image, in_size):
    w, h, = image.size
    if w > h:
        image = image.resize((in_size / h * w, in_size))
    else:
        image = image.resize((in_size, in_size / w * h))
    return image


def crop_square(im):
    width, height = im.size
    in_size = max(im.size)
    left = (width - in_size) // 2
    top = (height - in_size) // 2
    right = (width + in_size) //2
    bottom = (height + in_size) //2    
    return im.crop((left, top, right, bottom))


def load_image(path, in_size, mean_image):
    image = Image.open(path)
    image = resize_max(image, in_size)
    image = crop_square(image)
    image = image.resize((in_size, in_size))
    image = numpy.asarray(image, dtype=numpy.float32)

    if len(image.shape) == 2: # grayscale
        image = numpy.dstack([image] * 3)
    elif image.shape[2] > 3:  # RGBA
        image = image[:,:,:3]

    image = image.transpose(2, 0, 1)[::-1] # BRG
    image -= mean_image
    return image


def grep_images(d):
    matches = []
    for root, dirnames, filenames in os.walk(d, followlinks=True):
        for e in ["jpg", "jpeg", "png", "gif", "bmp"]:
            for filename in fnmatch.filter(filenames, '*.' + e):
                matches.append(os.path.join(root, filename))
    return matches


def join_path(a, b):
    if a[-1] is not '/':
        a += '/'
    if b[0] is '.':
        b = b[1:]
    if b[0] is '/':
        b = b[1:]
    return a + b


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def save_features(dst_dir, path_list, features):
    for i, p in enumerate(path_list):        
        f = chainer.cuda.to_cpu(features.data[i])

        name, ext = os.path.splitext(p)
        p = join_path(dst_dir, name + ".csv")
        d = os.path.dirname(p)
        mkdir(d)
        numpy.savetxt(p, f)

