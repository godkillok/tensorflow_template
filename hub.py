import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd

import re
module_url = "https://tfhub.dev/google/universal-sentence-encoder/1" #@param ["https://tfhub.dev/google/universal-sentence-encoder/1", "https://tfhub.dev/google/universal-sentence-encoder-large/1"]
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

def parser(record):
    keys_to_features = {
        'image_raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label

def train_input_fn():
    '''
    训练输入函数，返回一个 batch 的 features 和 labels
    '''
    train_dataset = tf.data.TFRecordDataset(FLAGS.train_dataset)
    train_dataset = train_dataset.map(parsecr)
    # num_epochs 为整个数据集的迭代次数
    train_dataset = train_dataset.repeat(FLAGS.num_epochs)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()

    features, labels = train_iterator.get_next()
    return features, labels