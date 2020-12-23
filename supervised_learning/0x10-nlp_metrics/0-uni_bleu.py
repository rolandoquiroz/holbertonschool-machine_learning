#!/usr/bin/env python3
"""
module 0-uni_bleu contains function uni_bleu
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

    clipped_counter = {k: min(ct_translation_u.get(k, 0), res.get(k, 0))
                       for k in ct_translation_u}

    return clipped_counter


def modified_precision(translation_u, list_of_reference_u, ngram=1):
    """
    Function that calculates modified precision
    """
    ct_clip = count_clip_ngram(translation_u, list_of_reference_u, ngram)
    ct = count_ngram(translation_u, ngram)

    modified_precision_value = (sum(ct_clip.values()) /
                                float(max(sum(ct.values()), 1)))

    return modified_precision_value


def closest_ref_length(translation_u, list_of_reference_u):
    """
    Determine the closest reference length from translation length
    """
    len_trans = len(translation_u)
    closest_ref_idx = np.argmin([abs(len(x) - len_trans)
                                 for x in list_of_reference_u])

    closest_reference_length = len(list_of_reference_u[closest_ref_idx])

    return closest_reference_length


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


def uni_bleu(references, sentence):
    """Function that calculates the unigram BLEU score for a sentence

    Arguments:
        references is a list of reference translations
            each reference translation is a list of the unigrams
            in the translation
        sentence is a list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    bp = brevity_penalty(sentence, references)
    mp = modified_precision(sentence, references)
    UNIBLEU = bp * mp

    return UNIBLEU
