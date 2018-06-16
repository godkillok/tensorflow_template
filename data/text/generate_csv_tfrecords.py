#!/usr/bin/env python

import tensorflow as tf
import os
import csv

# 将数据转化成对应的属性
def feature_auto(value):
  if isinstance(value,int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


  elif isinstance(value,str):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

  elif isinstance(value, float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)
  cout=0
  with open(input_filename, 'r+', newline='') as csv_file:
    reader = csv.reader(csv_file)
    for data in reader:



      text = str(data[0])
      label = str(data[1])

      # To show an example of embedding

      example = tf.train.Example(features=tf.train.Features(feature={
        'text': feature_auto(text),
        'label': feature_auto(label)
      }))
      if cout < 8:
        writer.write(example.SerializeToString())

  writer.close()
  print("Successfully convert {} to {}".format(input_filename,
                                               output_filename))


def main():
  current_path = os.getcwd()
  for filename in os.listdir(current_path):
    if filename.startswith("") and filename.endswith(".csv"):
      generate_tfrecords(filename, filename + ".tfrecords")


if __name__ == "__main__":
  main()
