#!/usr/bin/env python3
"""
module 5-convolve
contains function convolve_channels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels

    Args:
        images: `numpy.ndarray` with shape (m, h, w)
            containing multiple grayscale images
            m: `int`, is the number of images
            h: `int`, is the height in pixels of the images
            w: `int`, is the width in pixels of the images
            c: `int`, is the number of channels in the image
        kernels: `numpy.ndarray` with shape (kh, kw, c, nc)
            containing the kernel for the convolution
            kh: `int`, is the height of the kernel
            kw: `int`, is the width of the kernel
            nc: `int`, is the number of kernels
        padding: `tuple` of (ph, pw), ‘same’, or ‘valid’
            if `tuple`:
                ph: `int` is the padding for the height of the image
                pw: `int` is the padding for the width of the image
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
        stride is a tuple of (sh, sw)
            sh: `int`, is the stride for the height of the image
            sw: `int`, is the stride for the width of the image

    Returns:
        output: `numpy.ndarray` containing the convolved images
    """
    c = images.shape[3]
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw, nc = kernels.shape[0], kernels.shape[1], kernels.shape[3]
    sh, sw = stride[0], stride[1]

    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]

    if padding == "same":
        ph = ((h - 1)*sh + kh - h)//2 + 1
        pw = ((w - 1)*sw + kw - w)//2 + 1

    if padding == "valid":
        ph = 0
        pw = 0

    custom_padded_images = np.pad(images,
                                  pad_width=((0, 0),
                                             (ph, ph), (pw, pw),
                                             (0, 0)),
                                  mode='constant',
                                  constant_values=0)

    h_custom_padded = (custom_padded_images.shape[1] - kh)//sh + 1
    w_custom_padded = (custom_padded_images.shape[2] - kw)//sw + 1

    output = np.zeros((m, h_custom_padded, w_custom_padded, nc))

    for j in range(h_custom_padded):
        for i in range(w_custom_padded):
            for k in range(nc):
                output[:, j, i, k] = ((kernel[:, :, :, k] *
                                      custom_padded_images[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw,
                                      :]).sum(axis=(1, 2, 3)))
    return output
