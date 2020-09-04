#!/usr/bin/env python3
"""
module 6-pool
contains function pool
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs a convolution on images using multiple kernels

    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
            c: `int`, is the number of channels in the image
        kernel_shape is a tuple of (kh, kw) containing
            the kernel shape for the pooling
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
        stride is a `tuple` of (sh, sw)
            sh: `int`, is the stride for the height of the image
            sw: `int`, is the stride for the width of the image
        mode: `str`, indicates the type of pooling
            max: indicates max pooling
            avg: indicates average pooling

    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    c = images.shape[3]
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]

    output_h = (h - kh)//sh + 1
    output_w = (w - kw)//sw + 1

    output = np.zeros((m, output_h, output_w, c))

    for j in range(output_h):
        for i in range(output_w):
            if mode == "max":
                output[:, j, i, :] = (np.max(images[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2)))
            if mode == "avg":
                output[:, j, i, :] = (np.mean(images[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2)))
    return output
