#!/usr/bin/env python3
"""
module 0-uni_bleu contains function uni_bleu
"""
import numpy as np


def uni_bleu(references, sentence):
    """Function that calculates the unigram BLEU score for a sentence

    Arguments:
        references is a list of reference translations
            each reference translation is a list of the words
            in the translation
        sentence is a list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
