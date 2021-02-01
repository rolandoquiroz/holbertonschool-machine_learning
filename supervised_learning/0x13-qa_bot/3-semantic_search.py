#!/usr/bin/env python3
"""Function semantic_search"""
from os import listdir
import tensorflow_hub as hub
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Function that performs semantic search on a corpus of documents.
    https://github.com/tensorflow/hub/blob/master/examples/colab/
    semantic_similarity_with_tf_hub_universal_encoder.ipynb

    Parameters
    ----------
    corpus_path : str
        the path to the corpus of reference documents on which to perform
        semantic search
    sentence : str
        the sentence from which to perform semantic search

    Returns
    -------
    reference : str
        the reference text of the document most similar to sentence
    """
    embed = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')

    articles = [sentence]
    for filename in listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        with open(f'{corpus_path}/{filename}',
                  mode='r', encoding='utf-8') as file:
            articles.append(file.read())

    embeddings = embed(articles)
    # The semantic similarity of two sentences can be trivially computed as
    # the inner product of the encodings
    corr = np.inner(embeddings, embeddings)
    closest = np.argmax(corr[0, 1:])
    reference = articles[closest + 1]

    return reference
