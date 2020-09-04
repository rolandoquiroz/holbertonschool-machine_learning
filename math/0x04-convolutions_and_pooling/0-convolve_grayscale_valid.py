#!/usr/bin/env python3
"""
module 0-convolve_grayscale_valid
contains function convolve_grayscale_valid
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
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
        a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    output_height = h - kh + 1
    output_width = w - kw + 1

    output = np.zeros((m, output_height, output_width))

    for i in range(output_width):
        for j in range(output_height):
            output[:, i, j] = (kernel * images[:, i: i + kh, j: j + kw]).sum(axis=(1, 2))
    return output
