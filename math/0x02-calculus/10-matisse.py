#!/usr/bin/env python3
"""10-matisse module

This file is executable and was interpreted/compiled on Ubuntu 16.04 LTS
using python3 (version 3.5.2) and pycodestyle style (version 2.5) was
used in.

It can also be imported as a module and contains the following functions:

    * summation_i_squared - Calculates the sum of i^2 as i goes from 1 to n
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
    if (type(poly) is not list):
        return None

    if poly is []:
        return None

    for i in range(len(poly)):
        if type(poly[i]) not in [int, float]:
            return None

    reversed_poly = []
    reversed_poly = poly[::-1]

    for element in reversed_poly:
        if element == 0:
            reversed_poly = reversed_poly[1:]
        else:
            break

    poly = reversed_poly[::-1]

    if len(poly) == 1:
        return [0]

    derivative_coefficients = []
    for i in range(1, len(poly)):
        derivative_coefficients.append(i*poly[i])
    return derivative_coefficients
