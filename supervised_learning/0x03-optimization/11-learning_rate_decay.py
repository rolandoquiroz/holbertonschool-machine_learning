#!/usr/bin/env python3
"""11-learning_rate_decay.py
contains the function learning_rate_decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy.
    the learning rate decay occurs in a stepwise fashion

    Args:
        alpha: `float`, the original learning rate
        decay_rate: `float`,the weight used to determine the rate at which
            alpha will decay
        global_step: `int`, the number of passes of gradient descent that
            have elapsed
        decay_step: `int`, the number of passes of gradient descent that
            should occur before alpha is decayed further

    Returns:
        alpha: `float`, the updated value for alpha.
    """
    alpha = alpha / (1 + decay_rate * int(global_step / decay_step))
    return alpha
