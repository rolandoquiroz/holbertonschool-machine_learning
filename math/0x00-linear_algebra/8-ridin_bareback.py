#!/usr/bin/env python3
'''mat_mul: Performs matrix multiplication'''


def mat_mul(mat1, mat2):
    '''mat_mul: Performs matrix multiplication'''
    if len(mat1[0]) == len(mat2):
        ans = []

        for _ in range(len(mat1)):
            ans.append([0]*len(mat2[0]))

        for row in range(len(mat1)):
            for col in range(len(mat2[0])):
                for k in range(len(mat2)):
                    ans[row][col] += mat1[row][k]*mat2[k][col]

        return ans
