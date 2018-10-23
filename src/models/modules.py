#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: modules.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import src.models.layers as L


def word_embedding_layer(inputs, vocab_size, embedding_dim, name='word_embedding_layer'):
    with tf.variable_scope(name):
        embedding = tf.get_variable(
            'embedding', [vocab_size, embedding_dim], dtype=tf.float32)
        return tf.nn.embedding_lookup(embedding, inputs)

def tile_for_num_beam(input_list, num_beam, bsize):
    input_list = tf.reshape(
        input_list, (-1, tf.shape(input_list)[-1])) # [batch * prev_beam, num_step]
    input_list = tf.contrib.seq2seq.tile_batch(
        input_list, multiplier=num_beam) # [batch * prev_beam * num_beam, num_step]
    input_list = tf.reshape(
        input_list, (bsize, -1, tf.shape(input_list)[-1])) #[bsize, num_beam * prev_beam, num_step]
    return input_list

def pick_by_indices(inputs, indices):
    # pick elements along the last axis by indices
    return tf.gather(inputs, indices=indices, axis=-1)

def find_top_likelihood_seq(rnn_logits, num_beam, stop_mask, logprob_list, bsize):
    rnn_softmax = tf.nn.softmax(rnn_logits) # [batch * prev_beam, vocab_size]
    # top_prob [batch * prev_beam, num_beam]
    # top_word_idx [batch * prev_beam, num_beam]
    top_prob, top_word_idx = tf.nn.top_k(rnn_softmax, num_beam)
    top_logprob = tf.log(top_prob) #[bsize * prev_beam, num_beam]
    logprob_list += tf.multiply(
        top_logprob,
        tf.cast(tf.reshape(stop_mask, (-1, num_beam)), tf.float32))
    logprob_list = tf.reshape(logprob_list, [bsize, -1]) #[bsize, prev_beam * num_beam]
    top_prob, top_idx = tf.nn.top_k(logprob_list, num_beam) #[bsize, num_beam]
    top_word_idx = tf.reshape(top_word_idx, [bsize, -1]) #[bsize, prev_beam * num_beam]

    return top_word_idx, logprob_list, top_idx

def update_tables(generate_word_list, prev_idx, top_idx, top_word_idx,
                  logprob_list, stop_mask, num_beam, stop_id, bsize):
    new_top_word_list = []
    new_logprob_list = []
    state_pick_id = []
    new_word_list = []
    pick_stop_mask = []

    prev_idx = tf.tile(prev_idx, (1, num_beam)) #[bsize * prev_beam, num_beam]
    prev_idx = tf.reshape(prev_idx, (bsize, -1)) #[bsize, prev_beam * num_beam]
    #[bsize, num_beam * prev_beam, num_step]
    generate_word_list = tile_for_num_beam(
        generate_word_list, num_beam, bsize)
    
    for batch_id in range(bsize):
        cur_idx = top_idx[batch_id]
        new_logprob_list.append(
            pick_by_indices(logprob_list[batch_id], cur_idx))
        state_pick_id.append(
            pick_by_indices(prev_idx[batch_id], cur_idx))
        pick_stop_mask.append(
            pick_by_indices(stop_mask[batch_id], cur_idx))
        
        cur_word = top_word_idx[batch_id]
        cur_w_pick = tf.gather(cur_word, indices=cur_idx, axis=-1)
        cur_word_list = generate_word_list[batch_id]
        prev_w_pick = tf.gather(cur_word_list, indices=cur_idx, axis=0)                
        new_word_list.append(tf.concat(
            (prev_w_pick, tf.expand_dims(cur_w_pick, axis=-1)),
            axis=-1))
        new_top_word_list.append(cur_w_pick)
        
    top_word_idx = tf.stack(new_top_word_list, axis=0) #[batch, num_beam]
    stop_mask = tf.logical_and(
        tf.stack(pick_stop_mask, axis=0),
        tf.not_equal(top_word_idx, stop_id))

    logprob_list = tf.stack(new_logprob_list, axis=0) #[batch, num_beam]
    logprob_list = tf.reshape(logprob_list, (-1, 1)) # [batch * prev_beam, 1]
    generate_word_list = tf.reshape(
        new_word_list, (bsize, num_beam, -1)) # [batch, num_beam, num_step]

    return logprob_list, generate_word_list, stop_mask, state_pick_id, top_word_idx

def update_rnn_state(state_pick_id, n_hidden, o_state):
    new_state = ()
    state_pick_id = tf.reshape(state_pick_id, (-1,))
    for cell_level_id in range(n_hidden):
        cur_state = o_state[cell_level_id]
        state_c = tf.gather(cur_state.c, indices=state_pick_id, axis=0)
        state_h = tf.gather(cur_state.h, indices=state_pick_id, axis=0)
        new_state += tf.nn.rnn_cell.LSTMStateTuple(c=state_c, h=state_h),
    return new_state
