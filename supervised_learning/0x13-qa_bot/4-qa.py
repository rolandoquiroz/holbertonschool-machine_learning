#!/usr/bin/env python3
"""Function question_answer"""
from os import listdir
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel
import numpy as np


def question_answer(question, reference):
    """
    Function that finds a snippet of text within a reference document
    to answer a question.

    Arguments
    ---------
    question : str
        The question to answer
    reference : str
        The reference document from which to find the answer

    Returns
    -------
    answer : str
        The answer for the question or None if no answer is found
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    question_tokens = tokenizer.tokenize(question)
    paragraph = f"<p>{reference}</p>"
    paragraph_tokens = tokenizer.tokenize(paragraph)
    tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + paragraph_tokens +\
        ["[SEP]"]

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] *\
        (len(paragraph_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])
    # using `[1:]` will enforce an answer.
    # `outputs[0][0][0]` is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer:
        return None
    return answer


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


def qa_bot(corpus_path):
    """
    Function that answers questions from multiple reference texts.

    Parameters
    ----------
    corpus_path : str
        the path to the corpus of reference documents
    """
    exits = ["exit", "quit", "goodbye", "bye"]
    while True:
        question = input("Q: ").lower().strip()
        if question in exits:
            print("A: Goodbye")
            exit()
        else:
            reference = semantic_search(corpus_path, question)
            answer = question_answer(question, reference)
            if answer is None:
                answer = 'Sorry, I do not understand your question.'
            print(f"A: {answer}")
