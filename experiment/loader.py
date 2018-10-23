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
from src.dataflow.coco import COCO

def load_coco(batch_size, rescale_size=224, shuffle=True):
    """ Load COCO data with caption

    Args:
        batch_size (int): batch size
        rescale_size (int): rescale image size
        shuffle (bool): whether shuffle data or not

    Retuns:
        COCO dataflow with image and caption
    """


    if platform.node() == 'arostitan':
        im_dir = '/home/qge2/workspace/data/dataset/COCO/train2014/'
        ann_dir = '/home/qge2/workspace/data/dataset/COCO/annotations_trainval2014/annotations/'
    elif platform.node() == 'Qians-MacBook-Pro.local':
        im_dir = '/Users/gq/workspace/Dataset/coco/train2014_small/'
        ann_dir = '/Users/gq/workspace/Dataset/coco/annotations_trainval2014/'
    elif platform.node() == 'aros04':
        im_dir = 'E:/Dataset/COCO/train2014/'
        ann_dir = 'E:/Dataset/COCO/annotations_trainval2014/annotations/'
    else:
        raise ValueError('Data path does not setup on this platform!')

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
    train_data = COCO(
        sample_range=[100, 30000],
        word_dict=word_dict['word_to_id'], 
        max_caption_len=60,
        im_dir=im_dir,
        ann_dir=ann_dir,
        shuffle=True,
        batch_dict_name=['image', 'caption'],
        pf_list=[preprocess_im, None])
    train_data.setup(epoch_val=0, batch_size=batch_size)

    valid_data = COCO(
        sample_range=[0, 100],
        word_dict=word_dict['word_to_id'], 
        max_caption_len=60,
        im_dir=im_dir,
        ann_dir=ann_dir,
        shuffle=shuffle,
        batch_dict_name=['image', 'caption'],
        pf_list=[preprocess_im, None])
    valid_data.setup(epoch_val=0, batch_size=batch_size)
    return train_data, valid_data

if __name__ == '__main__':
    # ann_dir = '/Users/gq/workspace/Dataset/coco/annotations_trainval2014/'
    # word_dict = np.load(
    #     os.path.join(ann_dir, 'word_dict.npy'), encoding='latin1').item()
    
    data, data2 = load_coco(2)
    id_to_word = data.id_to_word
    for i in range(50):
        batch_data = data2.next_batch_dict()
        # print(batch_data)

        cur_im = np.squeeze(batch_data['image'][0])
        print(batch_data['caption'])
        print([id_to_word[w_id] for w_id in batch_data['caption'][0]])
    # cur_im = ((cur_im + 1) * 255 / 2)
    # cur_im = cur_im.astype(np.uint8)

    # plt.figure()
    # plt.imshow(cur_im)
    # plt.show()

