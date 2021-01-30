#!/usr/bin/env python3
"""Function semantic_search"""
from os import listdir
import tensorflow_hub as hub
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Function that performs semantic search on a corpus of documents.

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
    articles = [sentence]
    for filename in listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        with open(f'{corpus_path}/{filename}', 'r', encoding='utf-8') as file:
            articles.append(file.read())
    embed = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')
    embeddings = embed(articles)
    corr = np.inner(embeddings, embeddings)
    closest = np.argmax(corr[0, 1:])
    return articles[closest + 1]
