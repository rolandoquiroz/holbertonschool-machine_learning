#!/usr/bin/env python3
"""
module 0-convolve_grayscale_valid
contains function convolve_grayscale_valid
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images custom padding

    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
        kernel: `numpy.ndarray` with shape (kh, kw)
            containing the kernel for the convolution
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
        padding: `tuple` of (ph, pw)
            ph: `int` is the padding for the height of the image
            pw: `int` is the padding for the width of the image

    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph, pw = padding[0], padding[1]

    h_custom_padded = h + 2*ph - kh + 1
    w_custom_padded = w + 2*pw - kw + 1

    custom_padded_images = np.pad(images,
                                  pad_width=((0, 0), (ph, ph), (pw, pw)),
                                  mode='constant',
                                  constant_values=0)

    output = np.zeros((m, h_custom_padded, w_custom_padded))

    for j in range(h_custom_padded):
        for i in range(w_custom_padded):
            output[:, j, i] = ((kernel * custom_padded_images[:,
                               j: j + kh, i: i + kw]).sum(axis=(1, 2)))
    return output
