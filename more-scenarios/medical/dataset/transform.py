import numpy as np
import random
import torch
from PIL import ImageFilter
from scipy import ndimage


def random_rot_flip(img, mask):
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    if mask is not None:
        mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    if mask is not None:
        mask = np.flip(mask, axis=axis).copy()
    return img, mask


def random_rotate(img, mask):
    angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=0, reshape=False)
    if mask is not None:
        mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def obtain_cutmix_box_3d(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size[0] * img_size[1] * img_size[2]
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_d = int(np.cbrt(size * ratio))
        cutmix_w = int(np.cbrt (size / ratio))
        cutmix_h = int(np.cbrt(size / ratio))
        d = np.random.randint(0, img_size[0])
        w = np.random.randint(0, img_size[1])
        h = np.random.randint(0, img_size[2])

        if w + cutmix_w <= img_size[0] and h + cutmix_h <= img_size[1] and d + cutmix_d <= img_size[2]:
            break

    mask[ d:d+cutmix_d,w:w + cutmix_w, h:h + cutmix_h] = 1
    return mask

