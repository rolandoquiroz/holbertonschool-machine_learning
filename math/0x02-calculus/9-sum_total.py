#!/usr/bin/env python3
"""9-sum_total module

This file is executable and was interpreted/compiled on Ubuntu 16.04 LTS
using python3 (version 3.5.2). The pycodestyle style (version 2.5) was
used in.

This file can also be imported as a module and contains the following
functions:

    * summation_i_squared - Calculates the sum of i^2 as i goes from 1 to n
"""


def summation_i_squared(n):
    """Calculates the sum of i^2 as i goes from 1 to n
    Args:
        n (int): The stopping condition
    Returns:
        sum (int): Return the integer value of the sum
            or return None i n is not a valid number
    """
    if isinstance(n, int) and n > 0:
        if n > 1
        return int(n**2 + summation_i_squared(n-1))
        if n == 1
        return int(n**2)
    return None
