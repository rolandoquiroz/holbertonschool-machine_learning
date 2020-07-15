#!/usr/bin/env python3
'''cat_matrices2D: Concatenates two matrices along a specific axis'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''cat_matrices2D: Concatenates two matrices along a specific axis'''
    if axis == 0:
        new_mat = mat1[:]
        new_mat.append(mat2[0])
    return new_mat
    

