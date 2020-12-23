#!/usr/bin/env python3
"""
module 1-ngram_bleu contains function ngram_bleu
"""
import numpy as np


def count_ngram(translation_u, ngram=1):
    """
    Function that counts n-grams in a sentence
    """
    tokens = zip(*[translation_u[i:] for i in range(ngram)])
    ngrams = [" ".join(token) for token in tokens]

    ngram_counter = {}
    for n_gram in ngrams:
        if n_gram not in ngram_counter:
            ngram_counter[n_gram] = ngrams.count(n_gram)

    return ngram_counter


def count_clip_ngram(sentence, references, ngram=1):
    """
    Function that counts clipped ngrams
    """
    clipped = {}
    sentence_ngrams_cntr = count_ngram(sentence, ngram)

    for reference in references:
        reference_ngrams_counter = count_ngram(reference, ngram)
        for n_gram in reference_ngrams_counter:
            if n_gram in clipped:
                clipped[n_gram] = max(reference_ngrams_counter[n_gram],
                                      clipped[n_gram])
            else:
                clipped[n_gram] = reference_ngrams_counter[n_gram]

    clipped_ngrams_counter = {n_gram: min(sentence_ngrams_cntr.get(n_gram, 0),
                              clipped.get(n_gram, 0))
                              for n_gram in sentence_ngrams_cntr}

    return clipped_ngrams_counter


def modified_precision(sentence, references, ngram=1):
    """
    Function that calculates modified precision
    """
    clipped_ngrams_cntr = count_clip_ngram(sentence, references, ngram)
    ngram_cntr = count_ngram(sentence, ngram)

    modified_precision_value = (sum(clipped_ngrams_cntr.values()) /
                                float(max(sum(ngram_cntr.values()), 1)))

    return modified_precision_value


def closest_reference_length(sentence, references):
    """
    Determine the closest reference length from translation length
    """
    sentence_length = len(sentence)
    closest_reference_index = np.argmin([abs(len(reference) - sentence_length)
                                         for reference in references])

    closest_reference_length = len(references[closest_reference_index])

    return closest_reference_length


def brevity_penalty(sentence, references):
    """
    Calculates brevety penalty
    """
    c = len(sentence)
    r = closest_reference_length(sentence, references)

    if c > r:
        return 1
    else:
        return np.exp(1 - r / c)


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
    bp = brevity_penalty(sentence, references)
    mp = modified_precision(sentence, references, n)
    BLEU = bp * mp

    return BLEU
