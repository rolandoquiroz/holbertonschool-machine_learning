#!/usr/bin/env python3
"""
module 2-cumulative_bleu contains function cumulative_bleu
"""
import numpy as np


def count_ngram(translation_u, ngram=1):
    """Function that counts n-grams in a sentence

        Arguments:
            sentence : a list containing the model proposed sentence
            ngram: is the size of the n-grams to be counted in sentence

        Returns:
            ngram_counter: a dictionary containing ngram as key,
                and count as value
    """
    tokens = zip(*[translation_u[i:] for i in range(ngram)])
    ngrams = [" ".join(token) for token in tokens]

    ngram_counter = {}
    for n_gram in ngrams:
        if n_gram not in ngram_counter:
            ngram_counter[n_gram] = ngrams.count(n_gram)
    return ngram_counter


def count_clip_ngram(translation_u, list_of_reference_u, ngram=1):
    """
    Function that counts clipped ngrams
    """
    res = {}
    ct_translation_u = count_ngram(translation_u, ngram)

    for reference_u in list_of_reference_u:
        ct_reference_u = count_ngram(reference_u, ngram)
        for k in ct_reference_u:
            if k in res:
                res[k] = max(ct_reference_u[k], res[k])
            else:
                res[k] = ct_reference_u[k]

    clipped = {k: min(ct_translation_u.get(k, 0), res.get(k, 0))
               for k in ct_translation_u}

    return clipped


def closest_ref_length(translation_u, list_of_reference_u):
    """
    Determine the closest reference length from translation length
    """
    len_trans = len(translation_u)
    closest_ref_idx = np.argmin([abs(len(x) - len_trans)
                                 for x in list_of_reference_u])

    closest_reference_lenght = len(list_of_reference_u[closest_ref_idx])

    return closest_reference_lenght


def brevity_penalty(translation_u, list_of_reference_u):
    """
    Calculates brevety penalty
    """
    c = len(translation_u)
    r = closest_ref_length(translation_u, list_of_reference_u)

    if c > r:
        return 1
    else:
        return np.exp(1 - r/c)


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
    c = len(count_ngram(sentence, n))
    clipped = count_clip_ngram(sentence, references, n)
    clipped_count = sum(clipped.values())
    BLEU = np.exp(np.log(clipped_count / c))
    return BLEU


def cumulative_bleu(references, sentence, n):
    """Function that calculates the cumulative n-gram BLEU score for a sentence

    Arguments:
        references is a list of reference translations
            each reference translation is a list of the words
            in the translation
        sentence is a list containing the model proposed sentence
        n is the size of the largest n-gram to use for evaluation
        All n-gram scores are weighted evenly

    Returns:
        the cumulative n-gram BLEU score
    """
    ngram_bleu_scores = [0] * n
    for i in range(0, n):
        ngram_bleu_scores[i] = ngram_bleu(references, sentence, i + 1)

    geo_mean = np.exp(np.mean(np.log(ngram_bleu_scores)))
    bp = brevity_penalty(sentence, references)
    CUMULATIVE_BLEU = bp * geo_mean

    if CUMULATIVE_BLEU > 0.4:
        return round(CUMULATIVE_BLEU, 7)

    return CUMULATIVE_BLEU
