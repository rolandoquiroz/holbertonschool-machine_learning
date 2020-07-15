#!/usr/bin/env python3
'''cat_arrays: that concatenates two arrays'''


def cat_arrays(arr1, arr2):
    '''cat_arrays: that concatenates two arrays'''
    new_list = arr1.copy()
    new_list.extend(arr2)
    return (new_list)
