#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tfutils.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf


def apply_mask(input_matrix, mask):
    return tf.dynamic_partition(input_matrix, mask, 2)[1]