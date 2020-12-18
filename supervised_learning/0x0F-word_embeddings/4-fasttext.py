#!/usr/bin/env python3
"""
module 4-fasttext contains
function fasttext_model
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Function that creates and trains a genism fastText model

            Arguments:
            sentences (list): sentences to be trained on
            size (int): dimensionality of the embedding layer
            min_count (int): minimum number of occurrences of
                a word for use in training
            window (int): maximum distance between the current
                and predicted word within a sentence
            negative (int): size of negative sampling
            cbow (bool): determines the training type:
                True is for CBOW
                False is for Skip-gram
            iterations (int): iterations to train over
            seed (int): seed for the random number generator
            workers (int): number of worker threads to train the model

        Returns:
            the trained model
    """
    model = FastText(sentences=sentences, size=size, min_count=min_count,
                     window=window, negative=negative, sg=cbow,
                     iter=iterations, seed=seed, workers=workers)

    model.train(sentences=sentences, total_examples=model.corpus_count,
                epochs=model.iter)

    return model
