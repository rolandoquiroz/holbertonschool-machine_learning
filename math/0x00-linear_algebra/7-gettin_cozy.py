#!/usr/bin/env python3
'''cat_matrices2D: Concatenates two matrices along a specific axis'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''cat_matrices2D: Concatenates two matrices along a specific axis'''
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            new_mat = [i[:] for i in mat1]
            for j in range(len(mat2)):
                new_mat.append(mat2[j])
            return new_mat

    if axis == 1:
        if len(mat1) == len(mat2):
            new_mat = [i[:] for i in mat1]
            for j in range(len(mat2)):
                new_mat[j].extend(mat2[j])
            return new_mat
