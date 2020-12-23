#!/usr/bin/env python3
"""
module 0-uni_bleu contains function uni_bleu
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
    Return
    ----

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

    return {k: min(ct_translation_u.get(k, 0), res.get(k, 0))
            for k in ct_translation_u}


def closest_ref_length(translation_u, list_of_reference_u):
    """
    determine the closest reference length from translation length
    """
    len_trans = len(translation_u)
    closest_ref_idx = np.argmin([abs(len(x) - len_trans)
                                 for x in list_of_reference_u])
    return len(list_of_reference_u[closest_ref_idx])


def brevity_penalty(translation_u, list_of_reference_u):
    """
    Something
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
    references_lenghts = []
    clipped = {}
    for unigram in sentence:
        foo = []
        for reference in references:
            references_lenghts.append(len(reference))
            foo.append(reference.count(unigram))
        if max(foo) > 0:
            clipped[unigram] = max(foo)

    c = len(sentence)
    clipped_count = sum(clipped.values())
    closest_reference_index = np.argmin([abs(len(reference) - c)
                                         for reference in references])
    r = len(references[closest_reference_index])

    if c > r:
        bp = 1
    else:
        bp = np.exp(1 - float(r) / float(c))

    BLEU = bp * np.exp(np.log(clipped_count / c))

    return BLEU
