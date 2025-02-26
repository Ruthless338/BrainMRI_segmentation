import numpy
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padw = size - ow if ow < size else 0
        padh = size - oh if oh < size else 0
        # pad第二个参数填充顺序为（左，上，右，下），故前两个参数为0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
        return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


