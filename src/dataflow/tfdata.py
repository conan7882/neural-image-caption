#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tfdata.py
# Author: Qian Ge <geqian1001@gmail.com>


import tensorflow as tf
from src.dataflow.base import DataFlow
from src.utils.utils import assert_type


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class tfrecordData(DataFlow):
    def __init__(self,
                 tfname,
                 record_names,
                 record_parse_list,
                 record_types,
                 raw_types,
                 decode_fncs,
                 batch_dict_name,
                 shuffle=True,
                 data_shape=[],
                 feature_len_list=None):
        if not isinstance(tfname, list):
            tfname = [tfname]
        if not isinstance(record_parse_list, list):
            record_parse_list = [record_parse_list]
        if not isinstance(record_names, list):
            record_names = [record_names]
        if not isinstance(record_types, list):
            record_types = [record_types]
        for c_type in record_types:
            assert_type(c_type, tf.DType)
        if not isinstance(raw_types, list):
            raw_types = [raw_types]
        for raw_type in raw_types:
            assert_type(raw_type, tf.DType)
        if not isinstance(decode_fncs, list):
            decode_fncs = [decode_fncs]
        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]   
        assert len(record_types) == len(record_names)
        assert len(record_types) == len(batch_dict_name)
        assert len(record_types) == len(record_parse_list)

        if feature_len_list is None:
            feature_len_list = [[] for i in range(0, len(record_names))]
        elif not isinstance(feature_len_list, list):
            feature_len_list = [feature_len_list]
        self._feat_len_list = feature_len_list
        if len(self._feat_len_list) < len(record_names):
            self._feat_len_list.extend([[] for i in range(0, len(record_names) - len(self._feat_len_list))])

        self.record_names = record_names
        self.record_types = record_types
        self.raw_types = raw_types
        self.decode_fncs = decode_fncs
        self.data_shape = data_shape
        self._tfname = tfname
        self._parse_list = record_parse_list

        self._shuffle = shuffle
        self._batch_dict_name = batch_dict_name

        self.setup_decode_data()
        self.setup(epoch_val=0, batch_size=1)

    def next_batch(self):
        sess = tf.get_default_session()
        batch_data = sess.run(self._data)

        self._batch_step += 1
        if self._batch_step % self._step_per_epoch == 0:
            self._epochs_completed += 1
        return batch_data

    def next_batch_dict(self):
        sess = tf.get_default_session()
        batch_data = sess.run(self._data)
        self._batch_step += 1
        if self._batch_step % self._step_per_epoch == 0:
            self._epochs_completed += 1
        batch_dict = {name: data for name, data in zip(self._batch_dict_name, batch_data)}
        return batch_dict

    def before_read_setup(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord)

    def after_reading(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

    def set_epochs_completed(self, val):
        self._set_epochs_completed(val)

    def _set_epochs_completed(self, val):
        self._epochs_completed  = val
        self._batch_step = 0

    def _set_batch_size(self, batch_size):
        self._batch_size = batch_size
        self.updata_data_op(batch_size)
        self.updata_step_per_epoch(batch_size)

    def updata_step_per_epoch(self, batch_size):
        self._step_per_epoch = int(self.size() / batch_size)

    def updata_data_op(self, batch_size):
        try:
            if self._shuffle is True:
                self._data = tf.train.shuffle_batch(
                    self._decode_data,
                    batch_size=batch_size,
                    capacity=batch_size * 4,
                    num_threads=2,
                    min_after_dequeue=batch_size * 2)
            else:
                print('***** data is not shuffled *****')
                self._data = tf.train.batch(
                    self._decode_data,
                    batch_size=batch_size,
                    capacity=batch_size,
                    num_threads=1,
                    allow_smaller_final_batch=False)
            # self._data = self._decode_data[0]
            # print(self._data)
        except AttributeError:
            pass

    def setup_decode_data(self):    
        feature = {}
        for record_name, r_type, cur_size, parse_fnc in zip(self.record_names, self.record_types, self._feat_len_list, self._parse_list):
            # feature[record_name] = tf.FixedLenFeature(cur_size, r_type)
            feature[record_name] = parse_fnc(cur_size, r_type)
            

        # filename_queue = tf.train.string_input_producer(self._tfname, num_epochs=n_epoch)
        filename_queue = tf.train.string_input_producer(self._tfname)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        
        features = tf.parse_single_example(serialized_example, features=feature)
        # print(features)

        # for example in tf.python_io.tf_record_iterator('E:/Dataset/COCO/tfrecord/coco_caption_train1.tfrecords'):
        #     result = tf.train.Example.FromString(example)
        #     print(result)


        decode_data = [decode_fnc(features[record_name], raw_type)
                       for decode_fnc, record_name, raw_type
                       in zip(self.decode_fncs, self.record_names, self.raw_types)]
        for idx, c_shape in enumerate(self.data_shape):
            if c_shape:
                decode_data[idx] = tf.reshape(decode_data[idx], c_shape)
        self._decode_data = decode_data

        try:
            self._set_batch_size(batch_size=self._batch_size)
        except AttributeError:
            self._set_batch_size(batch_size=1)

    def size(self):
        try:
            return self._size
        except AttributeError:
            self._size = sum(1 for f in self._tfname for _ in tf.python_io.tf_record_iterator(f))
            return self._size


class Bottleneck2TFrecord(object):
    def __init__(self, record_feat_name):
        self._feat_name = record_feat_name

    def write(self, tfname, get_feat_fnc, dataflow,
              record_dataflow_keys=[], record_dataflow_tfnames=[],
              convert_fncs=[]):

        if not isinstance(record_dataflow_tfnames, list):
            record_dataflow_tfnames = [record_dataflow_tfnames]
        if not isinstance(convert_fncs, list):
            convert_fncs = [convert_fncs]
        if not isinstance(record_dataflow_keys, list):
            record_dataflow_keys = [record_dataflow_keys]
        assert len(convert_fncs) == len(record_dataflow_tfnames)
        assert len(record_dataflow_keys) == len(record_dataflow_tfnames)

        dataflow.setup(epoch_val=0, batch_size=1)
        tfrecords_filename = tfname
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            cnt = 0
            while dataflow.epochs_completed < 1:
                print('Writing data {}...'.format(cnt))
                batch_data = dataflow.next_batch_dict()

                feat = get_feat_fnc(sess, batch_data)
                bsize = len(feat)

                for idx in range(bsize):
                    feature = {}
                    for record_name, c_fnc, key in zip(record_dataflow_tfnames, convert_fncs, record_dataflow_keys):
                        # print(batch_data[key][idx])
                        feature[record_name] = c_fnc(batch_data[key][idx])
                    # for record_name, feat in zip(self._w_f_names, feats):
                    feature[self._feat_name] = float_feature(feat[idx].reshape(-1).tolist())
                    # print(feature)

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

                cnt += 1

        writer.flush()
        writer.close()






