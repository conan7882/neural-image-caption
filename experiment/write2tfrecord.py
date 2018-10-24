#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: write2tfrecord.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import skimage.transform
import numpy as np
import tensorflow as tf
sys.path.append('../')
import loader as loader
from src.dataflow.coco import COCO
import src.models.inception_module as inception_module
from src.dataflow.tfdata import Bottleneck2TFrecord, int64_feature, bytes_feature, tfrecordData


if platform.node() == 'arostitan':
    CNN_PATH = '/home/qge2/workspace/data/pretrain/inception/googlenet.npy'
    SAVE_PATH = '/home/qge2/workspace/data/dataset/COCO/tfdata/'
elif platform.node() == 'Qians-MacBook-Pro.local':
    CNN_PATH = '/Users/gq/workspace/Dataset/pretrained/googlenet.npy'
    SAVE_PATH = '/Users/gq/workspace/Dataset/coco/tfrecord/'
elif platform.node() == 'aros04':
    SAVE_PATH = 'E:/Dataset/COCO/tfrecord/'
    CNN_PATH = 'E:/Dataset/pretrained/googlenet.npy'
else:
    raise ValueError('Data path does not setup on this platform!')

def read(tfrecord_filename, is_all_reference=False):
    data_path = os.path.join(SAVE_PATH, '{}.tfrecords'.format(tfrecord_filename))

    if is_all_reference:
        data = tfrecordData(tfname=data_path,
                            record_names=['inception_feat', 'filename'],
                            record_parse_list=[tf.FixedLenFeature, tf.FixedLenFeature],
                            record_types=[tf.float32, tf.string],
                            raw_types=[tf.float32, tf.string],
                            decode_fncs=[tf.cast, tf.cast],
                            batch_dict_name=['inception_feat', 'filename'],
                            feature_len_list=[[50176], []],
                            data_shape=[[7, 7, 1024], []],
                            shuffle=True)

    else:
        data = tfrecordData(tfname=data_path,
                            record_names=['inception_feat', 'caption', 'o_len'],
                            record_parse_list=[tf.FixedLenFeature, tf.FixedLenFeature, tf.FixedLenFeature],
                            record_types=[tf.float32, tf.int64, tf.int64],
                            raw_types=[tf.float32, tf.int64, tf.int64],
                            decode_fncs=[tf.cast, tf.cast, tf.cast],
                            batch_dict_name=['inception_feat', 'caption', 'o_len'],
                            feature_len_list=[[50176], [61]],
                            data_shape=[[7, 7, 1024], [61]],
                            shuffle=True)

    data.setup(epoch_val=0, batch_size=2)
    word_to_id, id_to_word = loader.load_coco_word_dict()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        data.before_read_setup()
        cnt = 0

        batch_data = data.next_batch_dict()

        if is_all_reference:
            print(batch_data['filename'])
        else:
            print(np.array(batch_data['o_len']))
            print(np.array(batch_data['caption']))
            print([id_to_word[w_id] for w_id in batch_data['caption'][0]])
        # while dataflow.epochs_completed < 1:
        #     print('Writing data {}...'.format(cnt))
        #     batch_data = dataflow.next_batch_dict()

        #     feats = []
        #     for feat_op, feed_plh_key, net_input_dict in zip(feat_ops, feed_plh_keys, net_input_dicts):
        #         feed_dict = {net_input_dict[key]: batch_data[key] for key in feed_plh_key}
        #         feats.append(sess.run(feat_op, feed_dict=feed_dict))

        #     feature = {}
        #     for record_name, convert_fnc, key in zip(record_dataflow_names, c_fncs, record_dataflow_keys):
        #         feature[record_name] = convert_fnc(batch_data[key][0])

        #     for record_name, feat in zip(record_net_names, feats):
        #         feature[record_name] = float_feature(feat.reshape(-1).tolist())


        #     example = tf.train.Example(features=tf.train.Features(feature=feature))
        #     writer.write(example.SerializeToString())
        #     cnt += 1
        data.after_reading()

def write(save_file_name, load_range, is_all_reference=False):
    image = tf.placeholder(
        tf.float32, [None, 224, 224, 3], name='image')
    pretrained_dict = np.load(
        CNN_PATH, encoding='latin1').item()
    inception_feat = inception_module.Inception_conv_embedding(
        inputs=image,
        pretrained_dict=pretrained_dict,
        name='Inception_conv')
    # print(inception_feat)

    def eval_inception_feat(sess, batch_data):
        feat = sess.run(
            inception_feat,
            feed_dict={image: batch_data['image']})
        return feat

    save_tfrecord_path = os.path.join(SAVE_PATH, '{}.tfrecords'.format(save_file_name))
    train_data = loader.load_coco(
        batch_size=1, load_range=load_range,
        rescale_size=224, shuffle=False,
        pad_with_max_len=True, is_all_reference=is_all_reference)
    tfwriter = Bottleneck2TFrecord('inception_feat')

    if is_all_reference:
        tfwriter.write(
            tfname=save_tfrecord_path,
            get_feat_fnc=eval_inception_feat,
            dataflow=train_data,
            record_dataflow_keys=['filename'],
            record_dataflow_tfnames=['filename'],
            convert_fncs=[bytes_feature])
    else:
        tfwriter.write(
            tfname=save_tfrecord_path,
            get_feat_fnc=eval_inception_feat,
            dataflow=train_data,
            record_dataflow_keys=['caption', 'o_len'],
            record_dataflow_tfnames=['caption', 'o_len'],
            convert_fncs=[int64_feature, int64_feature])

if __name__ == '__main__':
    # write(4, [20000, 25000])
    # write('test_2', load_range=[0, 2], is_all_reference=False)
    read('test', is_all_reference=True)
