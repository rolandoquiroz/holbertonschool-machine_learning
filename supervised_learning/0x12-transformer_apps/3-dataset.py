#!/usr/bin/env python3
"""class Dataset v 3.0"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """
        class Dataset constructor

        Arguments:
            batch_size: the batch size for training/validation
            max_len: the maximum number of tokens allowed
                per example sentence

        Attributes:
            data_train: Contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervided
            data_valid: Contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervided
            tokenizer_pt: Portuguese tokenizer created from the training set
            tokenizer_en: English tokenizer created from the training set
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)

        self.data_train, self.data_valid = examples['train'],\
            examples['validation']
        self.metadata = metadata
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        def filter_max_length(x, y, max_length=max_len):
            """
            Function that filter out all examples that have
            either sentence with more than max_len tokens
            """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()

        self.data_train = self.data_train.shuffle(
            self.metadata.splits['train'].num_examples)

        padded_shapes = ([None], [None])

        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=padded_shapes)

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=padded_shapes)

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
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)
        #              tfds.deprecated.text
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Method that encodes a translation into tokens.

        Arguments:
            pt: tf.Tensor
                the Portuguese sentence
            en: tf.Tensor
                the corresponding English sentence
            The tokenized sentences should include
                the start and end of sentence tokens
            The start token should be indexed as vocab_size
            The end token should be indexed as vocab_size + 1

        Returns:
            pt_tokens, en_tokens
                pt_tokens: np.ndarray
                    the Portuguese tokens
                en_tokens: np.ndarray
                    the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Method that acts as a tensorflow wrapper
        for the encode instance method.
        """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
