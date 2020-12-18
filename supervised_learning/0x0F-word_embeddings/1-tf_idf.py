#!/usr/bin/env python3
"""
module 1-tf_idf contains
function tf_idf
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Function that creates a TF-IDF embedding

        Arguments:
            sentences (list) : sentences to analyze
            vocab (list): contains the vocabulary words to use for the analysis
                If None, all words within sentences should be used

        Returns:
            embeddings, features
            embeddings (numpy.ndarray): shape (s, f)
                containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
            features (list): features used for embeddings
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features
