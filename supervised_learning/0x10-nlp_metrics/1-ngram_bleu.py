#!/usr/bin/env python3
"""
module 1-ngram_bleu contains function ngram_bleu
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence

    Arguments:
        references is a list of reference translations
            each reference translation is a list of the words
            in the translation
        sentence is a list containing the model proposed sentence
        n is the size of the largest n-gram to use for evaluation

    Returns:
        the n-gram BLEU score
    """
