#!/usr/bin/env python3
"""7-early_stopping module
contains the function early_stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early.
        Early stopping should occur when the validation cost
        of the network has not decreased relative to the optimal
        validation cost by more than the threshold over a specific
        patience count.

    Args:
        cost: `float`, is the current validation cost of the neural network
        opt_cost: `float`, is the lowest recorded validation cost of
            the neural network
        threshold: `float`, is the threshold used for early stopping
        patience: `int`, is the patience count used for early stopping
        count: `int`, is the count of how long the threshold has not been met

    Returns:
        `tuple` a boolean of whether the network should be stopped early,
            afollowed by the updated count
    """
    if (cost < opt_cost - threshold):
        return False, 0
    else:
        count += 1
        if (count < patience):
            return False, count
        else:
            return True, count
