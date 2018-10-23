#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: metric.py
# Author: Qian Ge <geqian1001@gmail.com>


import numpy as np
import tensorflow as tf
from src.utils.tfutils import apply_mask

SMALL_NUM = 1e-8


def tflog2(inputs):
    return tf.log(inputs) / tf.log(2.)


def get_perplexity(logits, valid_mask):
    """
    Args:
        logits: [batch*n_step, n_class]
        mask: [batch, n_step]
    """
    prob = tf.nn.softmax(logits, axis=-1)
    mask = tf.reshape(valid_mask, [-1])
    entropy = -tf.reduce_sum(tf.multiply(prob, tflog2(prob + SMALL_NUM)), axis=-1)
    perplexity_list = tf.pow(2., entropy)

    return tf.reduce_mean(apply_mask(perplexity_list, mask))

def np_get_perplexity(batch_prob, batch_word, stop_id):
    """
    
    Args:
        batch_prob: [batch, num_word, num_verbose]
        batch_word: [batch, num_word]

    """
    batch_log2prob = np.log2(batch_prob + SMALL_NUM)
    n_words = 0
    perplexity_sum = 0
    for prob, log2prob, words in zip(batch_prob, batch_log2prob, batch_word):
        try:
            gen_len = list(words).index(stop_id) + 1
        except ValueError:
            gen_len = len(list(words))
        n_words += gen_len
        entropy = -np.sum(np.multiply(prob, log2prob), axis=-1)
        perplexity_list = np.power(2., entropy)[:gen_len]
        perplexity_sum += np.sum(perplexity_list)
    return 1. * perplexity_sum / n_words
