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
# import loader as loader
from src.dataflow.coco import COCO
import src.models.inception_module as inception_module
from src.dataflow.tfdata import Bottleneck2TFrecord, int64_feature, tfrecordData


if platform.node() == 'arostitan':
    CNN_PATH = '/home/qge2/workspace/data/pretrain/inception/googlenet.npy'
    SAVE_PATH = '/home/qge2/workspace/data/dataset/COCO/tfdata/'
# elif platform.node() == 'Qians-MacBook-Pro.local':
#     im_dir = '/Users/gq/workspace/Dataset/coco/train2014_small/'
#     ann_dir = '/Users/gq/workspace/Dataset/coco/annotations_trainval2014/'
elif platform.node() == 'aros04':
    SAVE_PATH = 'E:/Dataset/COCO/tfrecord/'
    CNN_PATH = 'E:/Dataset/pretrained/googlenet.npy'
else:
    raise ValueError('Data path does not setup on this platform!')


def parse_caption(inputs, shape):
    return tf.FixedLenSequenceFeature(inputs, shape, allow_missing=True)

def load_data():
    if platform.node() == 'arostitan':
        im_dir = '/home/qge2/workspace/data/dataset/COCO/train2014/'
        ann_dir = '/home/qge2/workspace/data/dataset/COCO/annotations_trainval2014/annotations/'
    elif platform.node() == 'Qians-MacBook-Pro.local':
        im_dir = '/Users/gq/workspace/Dataset/coco/train2014_small/'
        ann_dir = '/Users/gq/workspace/Dataset/coco/annotations_trainval2014/'
    elif platform.node() == 'aros04':
        im_dir = 'E:/Dataset/COCO/train2014_small/'
        ann_dir = 'E:/Dataset/COCO/annotations_trainval2014/annotations/'
    else:
        raise ValueError('Data path does not setup on this platform!')

    def preprocess_im(im):
        im = skimage.transform.resize(
            im, [224, 224, 3],
            mode='reflect',
            preserve_range=True)
        return im.astype(np.uint8)

    word_dict = np.load(
        os.path.join(ann_dir, 'word_dict.npy'), encoding='latin1').item()
    # ann_path = os.path.join(ann_dir, 'captions_train2014.json')
    train_data = COCO(
        sample_range=[15000, 20000],
        word_dict=word_dict['word_to_id'], 
        max_caption_len=60,
        pad_with_max_len=True,
        im_dir=im_dir,
        ann_dir=ann_dir,
        shuffle=False,
        batch_dict_name=['image', 'caption', 'o_len'],
        pf_list=[preprocess_im, None])
    train_data.setup(epoch_val=0, batch_size=1)

    return train_data

def read():
    train_data = load_data()
    data_path = os.path.join(SAVE_PATH, 'coco_caption_train1.tfrecords')
    # data = tfrecordData(data_path)
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

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        data.before_read_setup()
        cnt = 0

        batch_data = data.next_batch_dict()
        print(np.array(batch_data['o_len']))
        print(np.array(batch_data['caption']))

        print([train_data.id_to_word[w_id] for w_id in batch_data['caption'][0]])
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

def write():
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

    save_tfrecord_path = os.path.join(SAVE_PATH, 'coco_caption_train3.tfrecords')
    train_data = load_data()
    tfwriter = Bottleneck2TFrecord('inception_feat')
    tfwriter.write(
        tfname=save_tfrecord_path,
        get_feat_fnc=eval_inception_feat,
        dataflow=train_data,
        record_dataflow_keys=['caption', 'o_len'],
        record_dataflow_tfnames=['caption', 'o_len'],
        convert_fncs=[int64_feature, int64_feature])

if __name__ == '__main__':
    write()
    # read()