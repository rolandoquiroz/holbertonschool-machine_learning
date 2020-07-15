#!/usr/bin/env python3
'''matrix_transpose: Returns the transpose of a 2D matrix'''


def matrix_transpose(matrix):
    '''matrix_transpose: Returns the transpose of a 2D matrix'''
    return [list(i) for i in zip(*matrix)]
