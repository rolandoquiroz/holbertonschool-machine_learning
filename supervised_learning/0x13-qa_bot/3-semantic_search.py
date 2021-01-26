#!/usr/bin/env python3
"""Function semantic_search"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel


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
