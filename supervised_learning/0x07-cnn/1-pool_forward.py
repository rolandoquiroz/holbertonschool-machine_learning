#!/usr/bin/env python3
"""
1-pool_forward module
contains function pool_forward
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs a convolution on images using multiple kernels
    Args:
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
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
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]

    output_h = (h - kh)//sh + 1
    output_w = (w - kw)//sw + 1

    output = np.zeros((m, output_h, output_w, c))

    for j in range(output_h):
        for i in range(output_w):
            if mode == "max":
                output[:, j, i, :] = (np.max(A_prev[[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2))))
            if mode == "avg":
                output[:, j, i, :] = (np.mean(A_prev[[:,
                                      j*sh: j*sh + kh,
                                      i*sw: i*sw + kw],
                                      axis=(1, 2))))
    return output
