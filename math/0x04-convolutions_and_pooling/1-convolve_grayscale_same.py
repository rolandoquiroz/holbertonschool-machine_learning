#!/usr/bin/env python3
"""
module 1-convolve_grayscale_same
contains function convolve_grayscale_same
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a valid convolution on grayscale images

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
    Returns:
         containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    output_h = h - kh + 1
    output_w = w - kw + 1

    ph = max(kh // 2, (kh - 1) // 2)
    pw = max(kw // 2, (kw - 1) // 2)

    padded_images = np.pad(images, [(0, 0), (ph, ph), (pw, pw)], 'constant',
                           constant_values=0)

    output = np.zeros((m, output_h, output_w))

    for i in range(output_w):
        for j in range(output_h):
            output[:, j, i] = ((kernel * padded_images[:,
                               j: j + kh, i: i + kw]).sum(axis=(1, 2)))
    return output
