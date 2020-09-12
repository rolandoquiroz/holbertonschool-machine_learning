#!/usr/bin/env python3
"""
1-pool_forward module
contains function pool_forward
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network

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
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]

    dA_prev = np.zeros(A_prev.shape)

    for l in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):
                    h_start = i * sh
                    h_end = h_start + kh
                    w_start = j * sw
                    w_end = w_start + kw
                    if mode == 'max':
                        value = np.max(A_prev[l, h_start:h_end, w_start:w_end,
                                              c])
                        mask = np.where(A_prev[l, h_start:h_end, w_start:w_end,
                                               c] == value, 1, 0)
                        mask = mask * dA[l, i, j, c]
                    if mode == 'avg':
                        mask = np.ones(kernel_shape)*(dA[l, i, j, c]/(kh*kw))
                    dA_prev[l, h_start:h_end, w_start:w_end, c] += mask
    return dA_prev
