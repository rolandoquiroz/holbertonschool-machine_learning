#!/usr/bin/env python3
"""Function question_answer"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel


def question_answer(corpus_path):
    """
    Function that answers questions from multiple reference texts.

    Parameters
    ----------
    corpus_path : str
        the path to the corpus of reference documents
    """
