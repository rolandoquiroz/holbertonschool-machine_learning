#!/usr/bin/env python3
''' matrix_shape: Gets the shape of a matrix'''


def matrix_shape(matrix):
    ''' matrix_shape: Gets the shape of a matrix'''
    m_s = [len(matrix)]
    while isinstance(matrix[0], list):
        m_s.append(len(matrix[0]))
        matrix = matrix[0]
    return m_s
