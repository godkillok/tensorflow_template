# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sqlalchemy import create_engine,MetaData,Table
import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import codecs

import numpy as np
from six.moves import urllib

import tensorflow as tf

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

def read_data():
    engine = create_engine('postgresql://postgres:123@localhost/postgres', encoding='utf8')
    sql = '''select  tags    from postgres.lang.tags where tags!='';'''
    sql_result = engine.execute(sql).fetchall()
    all_lines = [v['tags'] for v in sql_result]
    return all_lines

vocabulary = read_data()
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000000


def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

with codecs.open('./metadata.tsv', 'w','utf8') as f:

  for i in range(vocabulary_size):
      print(i)
      f.write(reverse_dictionary[i] + '\n')


def generate_batch2(i,dictionary):
  print(i)
  engine = create_engine('postgresql://postgres:123@localhost/postgres', encoding='utf8')

  sql = '''select tags from postgres.lang.short_view_result where _id>={} and _id<{};'''.format(i, i + 10000)

  sql_result = engine.execute(sql).fetchall()
  output_filename="./tags_{}.tfrecords".format(i)
  all_lines = [v['tags'] for v in sql_result]
  writer = tf.python_io.TFRecordWriter(output_filename)
  for line in all_lines:
    data = line.split(",")

    for d1 in data:
      for d2 in data:
        d1,d2=d1.lower().replace('.',''),d2.lower().replace('.','')
        if d1!=d2 and d1!='' and d2!='':
          values=int(dictionary.get(d1,0))

          if values==0:
              print(d1)

          label =int(dictionary.get(d2,0))

          if label==0:
              print(d2)

          # Write each example one by one
          example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "values": tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
          }))

          writer.write(example.SerializeToString())

  writer.close()
for i in range(1, 555576, 10000):
    generate_batch2(i, dictionary)
# from multiprocessing import Process,Pool
# p = Pool(8)
# for i in range(1, 555576, 10000):
#   p.apply_async(generate_batch2, (i, dictionary,))  # 增加新的进程
# p.close()  # 禁止在增加新的进程
# p.join()
# translate(2)
print("pool process done")


