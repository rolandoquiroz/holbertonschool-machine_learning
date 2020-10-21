#!/usr/bin/env python3
"""
0-determinant module
contains determinant function and recursive_determinant function
"""


def recursive_determinant(M, det=0):
    """Calculates the determinant of a matrix
    Args:
        M (list): list of lists whose determinant should be calculated
    Returns:
        det (float): the determinant of matrix
    """
    if len(M) == 1:
        return M[0][0]
    if len(M) == 2:
        return M[0][0] * M[1][1] - M[1][0] * M[0][1]

    i = list(range(len(M)))
    for cur_col in i:
        copied_matrix = [row[:] for row in M]
        sub_M = copied_matrix[1:]
        j = len(sub_M)

        for i in range(j):
            sub_M[i] = sub_M[i][0:cur_col] + sub_M[i][cur_col+1:]

        sub_det = recursive_determinant(sub_M)
        det += ((-1) ** (cur_col % 2)) * M[0][cur_col] * sub_det

    return det


def determinant(matrix):
    """Calculates the determinant of a matrix
    Args:
        matrix (list): list of lists whose determinant should be calculated
    Returns:
        (float): the determinant of matrix
    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if any(type(row) is not list for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if any(len(matrix) != len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    return(recursive_determinant(matrix))
