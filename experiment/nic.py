#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

# import os
# 
# import platform
# import scipy.misc
# import skimage.transform
import sys
import argparse
import platform
import numpy as np
import tensorflow as tf

sys.path.append('../')
import loader as loader
from src.nets.nic import NIC


if platform.node() == 'arostitan':
    CNN_PATH = '/home/qge2/workspace/data/pretrain/inception/googlenet.npy'
    SAVE_PATH = '/home/qge2/workspace/data/out/nic/'
elif platform.node() == 'Qians-MacBook-Pro.local':
    CNN_PATH = '/Users/gq/workspace/Dataset/pretrained/googlenet.npy'
    # CNN_PATH = None
# elif platform.node() == 'aros04':
#     im_dir = 'E:/Dataset/COCO/train2014/'
#     ann_dir = 'E:/Dataset/COCO/annotations_trainval2014/annotations/'
else:
    raise ValueError('Data path does not setup on this platform!')




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model.')
    parser.add_argument('--generate', action='store_true',
                        help='Sampling from trained model')

    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Init learning rate')
    parser.add_argument('--keep_prob', type=float, default=1.,
                        help='keep_prob')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Init learning rate')
    parser.add_argument('--maxepoch', type=int, default=100,
                        help='Max iteration')

    parser.add_argument('--embedding', type=int, default=512,
                        help='')
    parser.add_argument('--hidden', type=int, default=[512],
                        help='')


    return parser.parse_args()

def train():
    FLAGS = get_args()
    train_data, valid_data = loader.load_coco(batch_size=FLAGS.bsize)
    valid_data.setup(epoch_val=0, batch_size=2)

    id_to_word = train_data.id_to_word
    word_to_id = train_data.word_dict
    vocab_size = len(train_data.id_to_word)

    train_model = NIC(n_channel=3,
                      vocab_size=vocab_size,
                      embedding_dim=FLAGS.embedding,
                      rnn_hidden_list=FLAGS.hidden,
                      start_word_id=word_to_id['START'],
                      stop_word_id=word_to_id['.'],
                      inception_path=CNN_PATH)
    train_model.create_train_model()

    generate_model = NIC(n_channel=3,
                      vocab_size=vocab_size,
                      embedding_dim=FLAGS.embedding,
                      rnn_hidden_list=FLAGS.hidden,
                      start_word_id=word_to_id['START'],
                      stop_word_id=word_to_id['.'],
                      bsize=2,
                      inception_path=CNN_PATH)
    generate_model.create_generate_model()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(SAVE_PATH)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(FLAGS.maxepoch):

            train_model.train_epoch(
                sess, train_data, init_lr=FLAGS.lr,
                keep_prob=0.5, summary_writer=writer)
            saver.save(sess, '{}epoch-{}'.format(SAVE_PATH, epoch_id))

            batch_data = valid_data.next_batch_dict()
            generate_text = sess.run(
                generate_model.layers['generate'],
                # [generate_model.test1, generate_model.test2],
                feed_dict={generate_model.image: batch_data['image']})
            for text in generate_text:
                print([id_to_word[w_id] for w_id in text])
            print('-- label --')
            for text in batch_data['caption']:
                print([id_to_word[w_id] for w_id in text])
            # print(test1)
            # print(list(test2))
            # break

if __name__ == '__main__':
    train()
