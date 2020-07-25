#!/usr/bin/env python3
"""10-matisse module

This file can be imported as a module and contains the following functions:

    * summation_i_squared - Calculates the derivative of a polynomial

This is an executable file and was interpreted/compiled on Ubuntu 16.04 LTS
using python3 (version 3.5.2) and pycodestyle style (version 2.5) was also
used in.
"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial
    Args:
        poly (list): Coefficients representing a polynomial:
            - The index of the list represents the power of x that the
            coefficient belongs to
            - Example: if [f(x) = x^3 + 3x +5] , poly is equal to
            [5, 3, 0, 1]

    Returns:
        (list): Coefficients representing the derivative of the polynomial
        [0]: If the derivative is 0
        None: If poly is not valid
    """
    if type(poly) is not list:
        return None
    if len(poly) == 0 or poly == []:
        return None
    if len(poly) == 1:
        if (type(poly[0]) in [int, float]):
            return [0]
    if len(poly) > 1:
        derivative_coefficients = []
        for i in range(len(poly)):
            if type(poly[i]) not in [int, float]:
                return None
            derivative_coefficients.append(i*poly[i])
    return derivative_coefficients[1:]
