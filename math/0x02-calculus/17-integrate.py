#!/usr/bin/env python3
"""17-integrate module

This file is executable and was interpreted/compiled on Ubuntu 16.04 LTS
using python3 (version 3.5.2) and pycodestyle style (version 2.5) was
used in.

It can also be imported as a module and contains the following functions:

    * poly_integral - Calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial
    Args:
        poly (list): Coefficients representing a polynomial:
            - The index of the list represents the power of x that the
            coefficient belongs to
            - Example: if [f(x) = x^3 + 3x +5] , poly is equal to
            [5, 3, 0, 1]
        c (int): Integration constant

    Returns:
        (list): Coefficients representing the integral of the polynomial
        None: If poly or C are not valid
    """
    if type(poly) is not list or type(C) is not int:
        return None
    if len(poly) == 0 or poly == []:
        return None
    if poly == [0]:
        return [C]
    integral_coefficients = []
    for i in range(len(poly)):
        if type(poly[i]) not in [int, float]:
            return None
        if poly[i] % (i+1) == 0:
            integral_coefficients.insert(i, int(poly[i]/(i+1)))
        else:
            integral_coefficients.insert(i, poly[i]/(i+1))
    integral_coefficients.insert(0, C)
    return integral_coefficients
