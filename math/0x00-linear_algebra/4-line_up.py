#!/usr/bin/env python3
'''add_arrays: Adds two arrays element-wise'''


def add_arrays(arr1, arr2):
    '''add_arrays: Adds two arrays element-wise'''
    if len(arr1) == len(arr2):
        return [i + j for i, j in zip(arr1, arr2)]
