#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gen_word_dict.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import json
import nltk
from collections import defaultdict, Counter
import numpy as np


DATA_PATH = 'E:/Dataset/COCO/annotations_trainval2014/annotations/'
DATA_PATH = '/Users/gq/workspace/Dataset/coco/annotations_trainval2014/'

def gen_dict(data_path):
    # reference code:
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L90
    with open(data_path) as f:
        data = json.load(f)

    anns = {}
    imgToID = {}
    imgToAnns = defaultdict(list)
    word_list = []
    for idx, ann in enumerate(data['annotations']):
        print(idx)

        tokenized_caption = nltk.word_tokenize(ann['caption'])
        print(len(tokenized_caption))
        # word_list.extend(tokenized_caption)
        # imgToAnns[ann['image_id']].append(tokenized_caption)
        # anns[ann['id']] = ann
        # if idx == 1000:
        #     break

    # for image in data['images']:
    #     imgToID[image['file_name']] = image['id'] 

    # print(data['images'][10])
    # print(imgToAnns[data['images'][100]['id']])
    # print(word_list)
    # counter = Counter(word_list)
    # count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # count_pairs = [c for c in count_pairs if c[1]>= 5 ]
    
    # words, _ = list(zip(*count_pairs))
    # word_to_id = dict(zip(words, range(len(words))))
    # print(word_to_id)

    # np.save(os.path.join(DATA_PATH, 'word_to_id_small.npy'), word_to_id)
    # data_dict = {'imgToAnns': imgToAnns, 'imgToID': imgToID}
    # np.save(os.path.join(ann_dir, 'annotations.npy'), data_dict)
    # np.save(os.path.join(DATA_PATH, 'imgToAnns.npy'), imgToAnns)

    # saved_dict = np.load(
    #     os.path.join(DATA_PATH, 'word_to_id.npy'),
    #     mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='latin1')

if __name__ == '__main__':
    ann_path = os.path.join(DATA_PATH, 'captions_train2014.json')
    gen_dict(ann_path)



