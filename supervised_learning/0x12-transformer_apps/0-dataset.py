#!/usr/bin/env python3
"""class Dataset"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """
        class Dataset constructor

        Attributes:
            data_train: Contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervided
            data_valid: Contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt: Portuguese tokenizer created from the training set
            tokenizer_en: English tokenizer created from the training set
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train',
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation',
                                    as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = tokenizer_pt, tokenizer_en

    def tokenize_dataset(self, data):
        """
        Method that creates sub-word tokenizers for our dataset

        Arguments:
            data: tf.data.Dataset whose examples are formatted
                as a tuple (pt, en)
                pt: tf.Tensor
                    the Portuguese sentence
                en: the tf.Tensor containing
                    the corresponding English sentence
            The maximum vocab size should be set to 2**15

        Returns:
            tokenizer_pt, tokenizer_en
                tokenizer_pt is the Portuguese tokenizer
                tokenizer_en is the English tokenizer
        """
        #              tfds.deprecated.text
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        #              tfds.deprecated.text
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en
