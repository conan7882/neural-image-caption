#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: coco.py
# Author: Qian Ge <geqian1001@gmail.com>

import re
import os
import json
from collections import defaultdict
import numpy as np
import tensorflow as tf
from src.dataflow.base import DataFlow
from src.utils.dataflow import get_file_list, load_image
import src.utils.utils as utils
from src.dataflow.tfdata import Bottleneck2TFrecord, int64_feature, tfrecordData

def identity(inputs):
    return inputs

class Inception_COCO(DataFlow):
    def __init__(self, word_dict, tf_dir, batch_dict_name,
                 stop_word='.', start_word='START', shuffle=True):
    
        self._batch_dict_name = batch_dict_name
        self.word_dict = word_dict
        self.n_words = len(word_dict)
        if 'UNK' not in self.word_dict:
            self.word_dict['UNK'] = self.n_words
            self.n_words += 1
        self._stop_word = stop_word
        if self._stop_word not in self.word_dict:
            self.word_dict[self._stop_word] = self.n_words
            self.n_words += 1
        self._start_word = start_word
        if self._start_word not in self.word_dict:
            self.word_dict[self._start_word] = self.n_words
            self.n_words += 1

        id_to_word = {}

        for key in word_dict:
            id_to_word[word_dict[key]] = key
        self.id_to_word = id_to_word

        self.tfdata = tfrecordData(
            tfname=tf_dir,
            record_names=['inception_feat', 'caption', 'o_len'],
            record_parse_list=[tf.FixedLenFeature, tf.FixedLenFeature, tf.FixedLenFeature],
            record_types=[tf.float32, tf.int64, tf.int64],
            raw_types=[tf.float32, tf.int64, tf.int64],
            decode_fncs=[tf.cast, tf.cast, tf.cast],
            batch_dict_name=['inception_feat', 'caption', 'o_len'],
            feature_len_list=[[50176], [61]],
            data_shape=[[7, 7, 1024], [61]],
            shuffle=shuffle)

    def next_batch(self):
        raw_batch_data = self.tfdata.next_batch()
        batch_caption = raw_batch_data[1]
        batch_o_len = raw_batch_data[2]

        max_len = np.amax(batch_o_len)
        truncate_caption = []
        caption_mask = []
        for caption, o_len in zip(batch_caption, batch_o_len):
            cur_caption = np.concatenate((caption[:max_len], [self.word_dict[self._stop_word]]), axis=0)
            truncate_caption.append(cur_caption)
            # remove first start
            cur_mask = [1 if i < o_len - 1 else 0 for i in range(max_len)]
            caption_mask.append(cur_mask)

        # for key in imgToAnns:
        #     cur_caption_list = imgToAnns[key]
        #     for c in cur_caption_list:
        #         len_c = len(c)
        #         if len_c > max_len:
        #             max_len = len_c
        # self.word_dict[self._stop_word]
        # print(np.array(truncate_caption))
        # print(np.array(caption_mask))
        batch_data = [np.array(raw_batch_data[0]), np.array(truncate_caption), np.array(caption_mask)]
        return batch_data

    def _set_epochs_completed(self, val):
        self.tfdata._set_epochs_completed(val)

    def _set_batch_size(self, batch_size):
        self.tfdata._set_batch_size(batch_size)

    def size(self):
        return self.tfdata.size()

    def get_batch_file_name(self):
        pass

    def before_read_setup(self):
        self.tfdata.before_read_setup()

    def after_reading(self):
        self.tfdata.after_reading()

    @property
    def epochs_completed(self):
        return self.tfdata._epochs_completed


class COCO(DataFlow):
    def __init__(self,
                 word_dict,
                 sample_range=None,
                 stop_word='.',
                 start_word='START',
                 max_caption_len=None,
                 pad_with_max_len=False,
                 im_dir='',
                 ann_dir='',
                 shuffle=True,
                 batch_dict_name=None,
                 pf_list=None):

        if pf_list is None:
            pf_list = [identity, identity]

        pf_list = utils.make_list(pf_list)
        if len(pf_list) == 1:
            pf_list.append(identity)
        assert len(pf_list) == 2

        def read_im(file_name):
            return load_image(file_name, read_channel=3,  pf=pf_list[0])

        def read_caption(file_name):
            return self._caption_dict[file_name]

        def o_caption_len(file_name):
            return [len(self._caption_dict[file_name])]

        self.word_dict = word_dict
        self.n_words = len(word_dict)
        if 'UNK' not in self.word_dict:
            self.word_dict['UNK'] = self.n_words
            self.n_words += 1
        self._stop_word = stop_word
        if self._stop_word not in self.word_dict:
            self.word_dict[self._stop_word] = self.n_words
            self.n_words += 1
        self._start_word = start_word
        if self._start_word not in self.word_dict:
            self.word_dict[self._start_word] = self.n_words
            self.n_words += 1

        id_to_word = {}
        for key in word_dict:
            id_to_word[word_dict[key]] = key
        self.id_to_word = id_to_word

        self.max_len = max_caption_len
        self._pad_max_len = pad_with_max_len
        self.sample_range = sample_range

        super(COCO, self).__init__(
            data_name_list=['.jpg', '.json', '.json'],
            data_dir=[im_dir, ann_dir, ann_dir],
            shuffle=shuffle,
            batch_dict_name=batch_dict_name,
            load_fnc_list=[read_im, read_caption, o_caption_len],
            )

    def _load_file_list(self, data_name_list, data_dir_list):
        self._file_name_list = [[] for i in range(3)]
        image_name_list = get_file_list(data_dir_list[0], data_name_list[0])
        if self.sample_range is not None:
            image_name_list = image_name_list[self.sample_range[0]: self.sample_range[1]]
            print('==== {} to {} samples are chosen. ===='
                  .format(self.sample_range[0], self.sample_range[1]))

        self._caption_dict = {}
        imgToAnns, imgToID = self._load_ann_data(data_dir_list[1])
        for image_name in image_name_list:
            drive, path_and_file = os.path.splitdrive(image_name)
            path, file_name = os.path.split(path_and_file)
            cur_caption_list = imgToAnns[imgToID[file_name]]
            word_id_list = []
            for idx, c in enumerate(cur_caption_list):
                self._file_name_list[0].append(image_name)
                self._file_name_list[1].append('{}_{}'.format(image_name, idx))
                c_id = [self.word_dict[w] if w in self.word_dict
                        else self.word_dict['UNK'] for w in c]
                c_id = [self.word_dict[self._start_word]] + c_id
                self._caption_dict['{}_{}'.format(image_name, idx)] = np.array(c_id)
                # word_id_list.append(np.array(c_id))

            # self._caption_dict[image_name] = word_id_list
        self._file_name_list[2] = self._file_name_list[1]
        for idx in range(len(self._file_name_list)):
            self._file_name_list[idx] = np.array(self._file_name_list[idx])
        if self._shuffle:
            self._suffle_file_list()

    def _load_ann_data(self, ann_dir):
        annotations = np.load(
            os.path.join(ann_dir, 'annotations.npy'),
            encoding='latin1').item()
        imgToAnns = annotations['imgToAnns']
        imgToID = annotations['imgToID']

        if self.max_len is None:
            max_len = 0
            for key in imgToAnns:
                cur_caption_list = imgToAnns[key]
                for c in cur_caption_list:
                    len_c = len(c)
                    if len_c > max_len:
                        max_len = len_c
            self.max_len = max_len

        return imgToAnns, imgToID

    def _batch_process(self, batch_data):
        batch_caption = batch_data[1]
        if self._pad_max_len:
            max_len = self.max_len
        else:
            max_len = 0
            for c in batch_caption:
                len_c = len(c)
                if max_len < len_c:
                    max_len = len_c

        batch_data[1] = np.array(
            [np.pad(c, (0, max_len + 1 - len(c)),
             'constant', constant_values=self.word_dict[self._stop_word])
            for c in batch_caption])

        return batch_data

