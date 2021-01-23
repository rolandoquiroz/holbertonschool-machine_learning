#!/usr/bin/env python3
"""Function answer_loop"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel


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

    if answer:
        return answer
    else:
        return None


def answer_loop(reference):
    """
    Function that that answers questions from a reference text.
    If the answer cannot be found in the reference text,
    respond with Sorry, I do not understand your question.

    Arguments
    ---------
    reference : str
        The reference document from which to find the answer
    """
    exits = ["exit", "quit", "goodbye", "bye"]

    while True:
        print("Q: ", end="")
        question = input().lower()

        if question in exits:
            print("A: Goodbye")
            exit()
        else:
            answer = question_answer(question, reference)
            if answer is None:
                answer = 'Sorry, I do not understand your question.'
            print(f"A: {answer}")
