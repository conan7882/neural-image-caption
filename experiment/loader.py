#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
# import scipy.misc
import skimage.transform
import numpy as np
# import matplotlib.pyplot as plt

sys.path.append('../')
from src.dataflow.images import Image
from src.dataflow.coco import COCO, Inception_COCO

if platform.node() == 'arostitan':
    im_dir = '/home/qge2/workspace/data/dataset/COCO/train2014/'
    tfrecord_dir = '/home/qge2/workspace/data/dataset/COCO/tfdata/'
    tfname_train = ['coco_caption_train{}.tfrecords'.format(i) for i in range(6)]
    tfname_valid = ['coco_caption_valid{}.tfrecords'.format(i) for i in range(1)]
    ann_dir = '/home/qge2/workspace/data/dataset/COCO/annotations_trainval2014/annotations/'
elif platform.node() == 'aros04':
    im_dir = 'E:/Dataset/COCO/train2014_small/'
    tfrecord_dir = 'E:/Dataset/COCO/tfrecord/'
    tfname = ['coco_caption_train1.tfrecords']
    ann_dir = 'E:/Dataset/COCO/annotations_trainval2014/annotations/'
elif platform.node() == 'Qians-MacBook-Pro.local':
    im_dir = '/Users/gq/workspace/Dataset/coco/train2014_small/'
    tfrecord_dir = '/Users/gq/workspace/Dataset/coco/tfrecord/'
    ann_dir = '/Users/gq/workspace/Dataset/coco/annotations_trainval2014/'
    tfname_train = ['test_2.tfrecords']
    tfname_valid = ['test.tfrecords']
else:
    raise ValueError('Data path does not setup on this platform!')


def load_coco_word_dict(stop_word='.', start_word='START', unknonw_word='UNK'):
    
    word_dict = np.load(
        os.path.join(ann_dir, 'word_dict.npy'), encoding='latin1').item()
    word_to_id = word_dict['word_to_id']

    n_words = len(word_to_id)
    if unknonw_word not in word_to_id:
        word_to_id[unknonw_word] = n_words
        n_words += 1
    if stop_word not in word_to_id:
        word_to_id[stop_word] = n_words
        n_words += 1
    if start_word not in word_to_id:
        word_to_id[start_word] = n_words
        n_words += 1

    id_to_word = {}
    for key in word_to_id:
        id_to_word[word_to_id[key]] = key
    
    return word_to_id, id_to_word

def load_test_im(batch_size=2, verbose=False):
    def preprocess_im(im):
        im = skimage.transform.resize(
            im, [224, 224, 3],
            mode='reflect',
            preserve_range=True)

        return im.astype(np.uint8)

    im_dir = '/home/qge2/workspace/data/test_im/'
    im_data = Image(im_name='.jpg',
                    data_dir=im_dir,
                    n_channel=3,
                    shuffle=False,
                    batch_dict_name=['image'],
                    pf_list=preprocess_im,
                    verbose=verbose)
    im_data.setup(epoch_val=0, batch_size=batch_size) 
    return im_data

def load_inception_coco(batch_size, shuffle=True):

    word_dict = np.load(
        os.path.join(ann_dir, 'word_dict.npy'), encoding='latin1').item()

    tf_dir = [os.path.join(tfrecord_dir, name) for name in tfname_train]
    train_data = Inception_COCO(
        tf_dir=tf_dir, word_dict=word_dict['word_to_id'],
        batch_dict_name=['image', 'caption', 'mask'],
        shuffle=shuffle)
    train_data.setup(epoch_val=0, batch_size=batch_size) 

    tf_dir = [os.path.join(tfrecord_dir, name) for name in tfname_valid]
    valid_data = Inception_COCO(
        tf_dir=tf_dir, word_dict=word_dict['word_to_id'],
        batch_dict_name=['image', 'caption', 'mask'],
        shuffle=False)
    valid_data.setup(epoch_val=0, batch_size=batch_size)   

    return train_data, valid_data

def load_coco(batch_size, load_range, rescale_size=224, shuffle=True,
              pad_with_max_len=False, is_all_reference=False):
    """ Load COCO data with caption

    Args:
        batch_size (int): batch size
        rescale_size (int): rescale image size
        shuffle (bool): whether shuffle data or not

    Retuns:
        COCO dataflow with image and caption
    """

    def preprocess_im(im):
        if rescale_size is not None:
            im = skimage.transform.resize(
                im, [rescale_size, rescale_size, 3],
                mode='reflect',
                preserve_range=True)
            # im = np.expand_dims(im, axis=-1)
        # im = im / 255. * 2. - 1.

        return im.astype(np.uint8)

    word_dict = np.load(
        os.path.join(ann_dir, 'word_dict.npy'), encoding='latin1').item()
    # ann_path = os.path.join(ann_dir, 'captions_train2014.json')
    data = COCO(
        sample_range=load_range,
        word_dict=word_dict['word_to_id'], 
        max_caption_len=60,
        pad_with_max_len=pad_with_max_len,
        im_dir=im_dir,
        ann_dir=ann_dir,
        shuffle=shuffle,
        batch_dict_name=['image', 'caption', 'o_len', 'filename'],
        pf_list=[preprocess_im],
        is_all_reference=is_all_reference)
    data.setup(epoch_val=0, batch_size=batch_size)

    # valid_data = COCO(
    #     sample_range=[40000, 40010],
    #     word_dict=word_dict['word_to_id'], 
    #     max_caption_len=60,
    #     pad_with_max_len=pad_with_max_len,
    #     im_dir=im_dir,
    #     ann_dir=ann_dir,
    #     shuffle=shuffle,
    #     batch_dict_name=['image', 'caption', 'o_len'],
    #     pf_list=[preprocess_im, None],
    #     is_all_reference=is_all_reference)
    # valid_data.setup(epoch_val=0, batch_size=batch_size)
    return data

if __name__ == '__main__':
    # ann_dir = '/Users/gq/workspace/Dataset/coco/annotations_trainval2014/'
    # word_dict = np.load(
    #     os.path.join(ann_dir, 'word_dict.npy'), encoding='latin1').item()
    
    # data, data2 = load_coco(2, pad_with_max_len=True)
    # id_to_word = data.id_to_word
    # for i in range(50):
    #     batch_data = data2.next_batch_dict()
    #     # print(batch_data)

    #     cur_im = np.squeeze(batch_data['image'][0])
    #     print(batch_data['caption'])
    #     print([id_to_word[w_id] for w_id in batch_data['caption'][0]])
    #     print(batch_data['o_len'])
    # cur_im = ((cur_im + 1) * 255 / 2)
    # cur_im = cur_im.astype(np.uint8)

    # plt.figure()
    # plt.imshow(cur_im)
    # plt.show()

    import tensorflow as tf
    data = load_inception_coco(batch_size=2, shuffle=True)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        data.before_read_setup()

        batch_data = data.next_batch_dict()

        print(batch_data['caption'])
        print([data.id_to_word[w_id] for w_id in batch_data['caption'][0]])

        data.after_reading()

