#!/usr/bin/env python

import tensorflow as tf
import os


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
  for line in open(input_filename, "r"):
    cout+=1

    data = line.split(",")
    age = int(data[0])
    workclass = str(data[1])
    fnlwgt = int(data[2])
    education = str(data[3])
    education_num = int(data[4])
    marital_status = str(data[5])
    occupation = str(data[6])
    relationship = str(data[7])
    race = str(data[8])
    gender = str(data[9])
    capital_gain = int(data[10])
    capital_loss = int(data[11])
    hours_per_week = int(data[12])
    native_country = str(data[13])
    income_bracket = str(data[14].replace('\n',''))
    # To show an example of embedding

    example = tf.train.Example(features=tf.train.Features(feature={
      'age': feature_auto(age),
      'workclass': feature_auto(workclass),
      'fnlwgt': feature_auto(fnlwgt),
      'education': feature_auto(education),
      'education_num': feature_auto(education_num),
      'marital_status': feature_auto(marital_status),
      'occupation': feature_auto(occupation),
      'relationship': feature_auto(relationship),
      'race': feature_auto(race),
      'gender': feature_auto(gender),
      'capital_gain': feature_auto(capital_gain),
      'capital_loss': feature_auto(capital_loss),
      'hours_per_week': feature_auto(hours_per_week),
      'native_country': feature_auto(native_country),
      'income_bracket': feature_auto(income_bracket)
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
