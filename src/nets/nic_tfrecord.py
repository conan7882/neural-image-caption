#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: nic_tfrecord.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

from src.nets.base import BaseModel
import src.utils.viz as viz
import src.models.layers as L
import src.models.modules as modules
import src.models.inception_module as inception_module


INIT_W = tf.keras.initializers.he_normal()
BN = False

class NIC_tfrecord(BaseModel):

    # def __init__(self, n_channel, n_class, pre_trained_path=None,
    #              bn=False, wd=0, trainable=True, sub_vgg_mean=True):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_list,
                 inception_path=None, n_channel=None,
                 start_word_id=None, stop_word_id=None,
                 bsize=None, num_beam=20, max_step=20):
        self._n_channel = n_channel
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._start_id = start_word_id
        self._stop_id = stop_word_id
        self._bsize = bsize
        self._n_beam = num_beam
        self._max_step = max_step

        if inception_path is not None:
            self._pretrained_dict = np.load(
                inception_path, encoding='latin1').item()

        if not isinstance(rnn_hidden_list, list):
            rnn_hidden_list = [rnn_hidden_list]
        self._hidden = rnn_hidden_list
        self._n_hidden = len(rnn_hidden_list)

        self.layers = {}

    def _create_train_input(self):
        self.image_feat = tf.placeholder(
            tf.float32, [None, 7, 7, 1024], name='image_feat')
        self.label = tf.placeholder(tf.int64, [None, None], 'label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.lr = tf.placeholder(tf.float32, name='lr')

    def create_train_model(self):
        self.set_is_training(is_training=True)
        self._create_train_input()
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            image_embedding, words_embedding = self._embedding_layer(self.image_feat, self.label, compute_feat=False)
            out_logits = self._rnn(image_embedding, words_embedding)

        self.layers['out_logits'] = out_logits

        self.epoch_id = 0
        self.global_step = 0
        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()

    def _create_generate_input(self):
        self.image = tf.placeholder(
            tf.float32, [None, 224, 224, self._n_channel], name='image')
        self.label = tf.placeholder(tf.int64, [None, None], 'label')
        self.keep_prob = 1.

    def create_generate_model(self):
        assert self._bsize is not None, 'bsize cannot be None for generate model'
        self.set_is_training(is_training=False)
        self._create_generate_input()
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            bsize = tf.shape(self.image)[0]
            start_word = [[self._start_id]]
            start_word = tf.tile(start_word, (bsize, 1))
            # start_word = [[self._start_id] for i in range(bsize)]
            image_embedding, words_embedding = self._embedding_layer(self.image, start_word)
            self.layers['generate'] = self._generate_sequence(image_embedding, words_embedding)
        
    def _embedding_layer(self, image_inputs, words_inputs, compute_feat=True):
        words_embedding = modules.word_embedding_layer(
            inputs=words_inputs,
            vocab_size=self._vocab_size,
            embedding_dim=self._embedding_dim,
            name='word_embedding_layer')

        if compute_feat:
            inception_conv_out = inception_module.Inception_conv_embedding(
                inputs=image_inputs,
                pretrained_dict=self._pretrained_dict,
                name='Inception_conv')
        else:
            inception_conv_out = image_inputs

        image_embedding = L.linear(
            out_dim=self._embedding_dim,
            inputs=inception_conv_out,
            init_w=INIT_W,
            wd=0,
            bn=BN,
            is_training=self.is_training,
            name='image_embedding_linear',
            # nl=tf.identity,
            add_summary=False)
        image_embedding = tf.reshape(image_embedding, (-1, 1, self._embedding_dim))

        return image_embedding, words_embedding

    def _rnn(self, image_embedding, words_embedding):
        with tf.variable_scope('rnn'):
            rnn_input = tf.concat((image_embedding, words_embedding), axis=1)
            rnn_out, _ = L.rnn_layer(
                inputs=rnn_input, hidden_size_list=self._hidden, init_state=None,
                is_training=self.is_training, keep_prob=self.keep_prob,
                name='rnn_layer')
            # remove the first and the last output
            rnn_out = rnn_out[:, 1:-1, :]
            rnn_out = tf.reshape(tf.concat(rnn_out, 1), [-1, self._hidden[-1]])

            out_logits = L.linear(
                out_dim=self._vocab_size,
                inputs=rnn_out,
                init_w=INIT_W,
                wd=0,
                bn=BN,
                is_training=self.is_training,
                name='rnn_output',
                # nl=tf.identity,
                add_summary=False)
            return out_logits

    def _rnn_generate(self, embedding, state):
        with tf.variable_scope('rnn'):
            rnn_out, state = L.rnn_layer(
                inputs=embedding, hidden_size_list=self._hidden, init_state=state,
                is_training=self.is_training, keep_prob=self.keep_prob,
                name='rnn_layer')
            step_len = tf.shape(rnn_out)[1]
            rnn_out = tf.reshape(tf.concat(rnn_out, 1), [-1, self._hidden[-1]])

            out_logits = L.linear(
                out_dim=self._vocab_size,
                inputs=rnn_out,
                is_training=self.is_training,
                name='rnn_output',
                # nl=tf.identity,
                add_summary=False)
            out_logits = tf.reshape(out_logits, [-1, step_len, self._vocab_size])
            return out_logits, state

    def _generate_sequence(self, image_embedding, start_embedding):
        # get initial state from image embedding
        _, state = self._rnn_generate(image_embedding, state=None)
        gen_word_list = self._beam_search(
            state=state, start_token=self._start_id, num_beam=self._n_beam,
            max_step=self._max_step)
        return gen_word_list

    def _beam_search(self, state, start_token, num_beam, max_step):
        """

        """
        bsize = self._bsize
        start_word = [[start_token]]
        start_word = tf.tile(start_word, (bsize, 1))
        generate_word_list = start_word
        stop_mask = [[True] for i in range(bsize)]
        prev_idx = [[i] for i in range(bsize)]

        top_word_idx = start_word
        logprob_list = tf.tile([[0.]], (bsize, 1)) # [batch * prev_beam, 1]
        for out_step in range(max_step):
            if out_step == max_step - 1:
                cur_num_beam = 1
            else:
                cur_num_beam = num_beam

            flatten_word = tf.reshape(top_word_idx, (-1, 1))
            word_embedding = modules.word_embedding_layer(
                inputs=flatten_word,
                vocab_size=self._vocab_size,
                embedding_dim=self._embedding_dim,
                name='word_embedding_layer')
            #[bsize, num_beam * prev_beam]
            stop_mask = tf.expand_dims(stop_mask, axis=-1)
            stop_mask = modules.tile_for_num_beam(
                stop_mask, cur_num_beam, bsize)
            stop_mask = tf.squeeze(stop_mask, axis=-1)

            rnn_out, state = self._rnn_generate(word_embedding, state)
            rnn_out = tf.squeeze(rnn_out, axis=1) # [batch * prev_beam, vocab_size]

            top_word_idx, logprob_list, top_idx = modules.find_top_likelihood_seq(
                rnn_out, cur_num_beam, stop_mask, logprob_list, bsize)
            logprob_list, generate_word_list, stop_mask, state_pick_id, top_word_idx = modules.update_tables(
                generate_word_list, prev_idx, top_idx, top_word_idx,
                logprob_list, stop_mask, cur_num_beam, self._stop_id, bsize)
            state = modules.update_rnn_state(state_pick_id, self._n_hidden, state)
            
            prev_idx = [[i] for i in range(bsize * cur_num_beam)]

        # self.test1 = state
        # self.test2 = top_word_idx
        # return generate_word_list
        return tf.squeeze(generate_word_list, axis=1)
                    
    def _get_loss(self):
        with tf.name_scope('loss'):
            label = self.label
            # remove the first START sign
            label = label[:, 1:]
            label = tf.reshape(label, [-1])

            self.test1 = label
            self.test2 = tf.argmax(self.layers['out_logits'], axis=-1)
            cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=self.layers['out_logits'], name='cross_entropy_loss')
            return tf.reduce_mean(cross_entropy_loss)

    def get_train_op(self):
        with tf.name_scope('train_op'):
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.get_loss(), tvars), 5.)
            grads = zip(grads, tvars)

            opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            train_op = opt.apply_gradients(grads, name='train_op')
            return train_op

    def train_epoch(self, sess, train_data, init_lr,
                    keep_prob=1.0, summary_writer=None):

        display_name_list = ['loss']
        cur_summary = None

        lr = init_lr

        cur_epoch = train_data.epochs_completed
        step = 0
        loss_sum = 0
        self.epoch_id += 1
        while cur_epoch == train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = train_data.next_batch_dict()
            _, loss = sess.run(
                [self.train_op, self.loss_op],
                feed_dict={self.label: batch_data['caption'],
                           self.image_feat: batch_data['image'],
                           self.keep_prob: keep_prob,
                           self.lr: lr})

            loss_sum += loss
            if step % 100 == 0:
            #     cur_summary = sess.run(
            #         self.train_summary_op, 
            #         feed_dict={self.real: im,
            #                    self.keep_prob: keep_prob,
            #                    self.random_vec: random_vec,
            #                    self.code_discrete: code_discrete,
            #                    self.discrete_label: discrete_label,
            #                    self.code_continuous: code_cont})

                viz.display(
                    self.global_step,
                    step,
                    [loss_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        # cur_summary = sess.run(
        #     self.train_summary_op, 
        #     feed_dict={self.real: im,
        #                self.keep_prob: keep_prob,
        #                self.random_vec: random_vec,
        #                self.code_discrete: code_discrete,
        #                self.discrete_label: discrete_label,
        #                self.code_continuous: code_cont})
        viz.display(
            self.global_step,
            step,
            [loss_sum],
            display_name_list,
            'train',
            summary_val=cur_summary,
            summary_writer=summary_writer)