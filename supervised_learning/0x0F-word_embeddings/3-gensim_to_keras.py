#!/usr/bin/env python3
"""
module 3-gensim_to_keras contains
function gensim_to_keras
"""
import tensorflow.keras as keras
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """Function that converts a gensim word2vec model to a keras Embedding
    layer

        Arguments:
            model is a trained gensim word2vec models

        Returns:
            the trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
