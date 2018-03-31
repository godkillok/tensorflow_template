#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import datetime
import logging
import os
import pprint

import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat


def define_flags():
    flags = tf.app.flags
    flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
    flags.DEFINE_string("mode", "train", "Support train, inference, savedmodel")
    flags.DEFINE_boolean("enable_benchmark", False, "Enable benchmark")
    flags.DEFINE_string("scenario", "classification",
                        "Support classification, regression")
    flags.DEFINE_integer("feature_size", 9, "Number of feature size")
    flags.DEFINE_integer("label_size", 2, "Number of label size")
    flags.DEFINE_string("train_file_format", "tfrecords",
                        "Support tfrecords, csv")
    flags.DEFINE_string("train_file", "./data/embeding/wide_deep_test.csv.tfrecords",
                        "Train files which supports glob pattern")
    flags.DEFINE_string("validate_file",
                        "./data/cancer/cancer_test.csv.tfrecords",
                        "Validate files which supports glob pattern")
    flags.DEFINE_string("inference_data_file", "./data/embeding/wide_deep_test.csv",
                        "Data file for inference")
    flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                        "Result file from inference")
    flags.DEFINE_string("optimizer", "adagrad",
                        "Support sgd, adadelta, adagrad, adam, ftrl, rmsprop")
    flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
    flags.DEFINE_string("model", "dnn",
                        "Support dnn, lr, wide_and_deep, customized, cnn")
    flags.DEFINE_string("dnn_struct", "128 32 8", "DNN struct")
    flags.DEFINE_integer("epoch_number", 1, "Number of epoches")
    flags.DEFINE_integer("batch_size", 3, "Batch size")
    flags.DEFINE_integer("validate_batch_size", 1024,
                         "Batch size for validation")
    flags.DEFINE_integer("batch_thread_number", 1, "Batch thread number")
    flags.DEFINE_integer("min_after_dequeue", 10, "Min after dequeue")
    flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization")
    flags.DEFINE_float("bn_epsilon", 0.001, "Epsilon of batch normalization")
    flags.DEFINE_boolean("enable_dropout", False, "Enable dropout")
    flags.DEFINE_float("dropout_keep_prob", 0.5, "Keep prob of dropout")
    flags.DEFINE_boolean("enable_lr_decay", False, "Enable learning rate decay")
    flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
    flags.DEFINE_integer("steps_to_validate", 10, "Steps to validate")
    flags.DEFINE_string("checkpoint_path", "./checkpoint/",
                        "Path for checkpoint")
    flags.DEFINE_string("output_path", "./tensorboard/", "Path for tensorboard")
    flags.DEFINE_string("model_path", "./model/", "Path of the model")
    flags.DEFINE_integer("model_version", 1, "Version of the model")
    FLAGS = flags.FLAGS
    return FLAGS


def assert_flags(FLAGS):
    if FLAGS.mode in ["train", "inference", "savedmodel"]:
        if FLAGS.scenario in ["classification", "regression"]:
            if FLAGS.train_file_format in ["tfrecords", "csv"]:
                if FLAGS.optimizer in [
                    "sgd", "adadelta", "adagrad", "adam", "ftrl", "rmsprop"
                ]:
                    if FLAGS.model in [
                        "dnn", "lr", "wide_and_deep", "customized", "cnn"
                    ]:
                        return

    logging.error("Get the unsupported parameters, exit now")
    exit(1)


def get_optimizer_by_name(optimizer_name, learning_rate):
    logging.info("Use the optimizer: {}".format(optimizer_name))
    if optimizer_name == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_name == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer_name == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer


def restore_from_checkpoint(sess, saver, checkpoint):
    if checkpoint:
        logging.info("Restore session from checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
        return True
    else:
        logging.warn("Checkpoint not found: {}".format(checkpoint))
        return False


def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]
    label = tf.feature_column.categorical_column_with_vocabulary_list(
        "income_bracket", [">50K", "<=50K"])
    return wide_columns, deep_columns, [label]


def build_model_columns2():
    """Builds a set of wide and deep feature columns."""
    # Continuous columns
    age = tf.feature_column.numeric_column('age')
    # education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th']))

    marital_status = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed']))

    relationship = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative']))

    workclass = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked']))

    # To show an example of hashing:
    occupation = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000), 8)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.indicator_column(tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000)),
        tf.feature_column.indicator_column(tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        # education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        workclass,
        education,
        marital_status,
        relationship,
        # To show an example of embedding
        occupation
    ]
    label = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
        "income_bracket", [">50K", "<=50K"]))
    return wide_columns, deep_columns, [label]


def read_and_decode_tfrecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    wide_columns, deep_columns,label_columns = build_model_columns2()
    # embedding_initializer=tf.contrib.framework.load_embedding_initializer(
    #       ckpt_path='C:/work/tensorflow_template/log/model.ckpt')

    from tensorflow.python import pywrap_tensorflow
    model_dir = 'C:/work/tensorflow_template/log/model.ckpt'
    checkpoint_path = model_dir
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    aa = reader.get_tensor('embeddings/Variable')

    examples = tf.parse_single_example(
        serialized_example,
        features={
            "education_num": tf.VarLenFeature(tf.int64),
            'workclass': tf.FixedLenFeature([], tf.string),
            'fnlwgt': tf.FixedLenFeature([], tf.int64),
            'education': tf.FixedLenFeature([], tf.string),

            'marital_status': tf.FixedLenFeature([], tf.string),
            'occupation': tf.FixedLenFeature([], tf.string),
            'relationship': tf.FixedLenFeature([], tf.string),
            'race': tf.FixedLenFeature([], tf.string),
            'gender': tf.FixedLenFeature([], tf.string),
            'capital_gain': tf.FixedLenFeature([], tf.int64),
            'capital_loss': tf.FixedLenFeature([], tf.int64),
            'hours_per_week': tf.FixedLenFeature([], tf.int64),
            'native_country': tf.FixedLenFeature([], tf.string),
            'age': tf.FixedLenFeature([], tf.int64),
            'income_bracket': tf.FixedLenFeature([], tf.string)

        })



    # batch_features = tf.train.shuffle_batch(
    #     examples,
    #     batch_size=FLAGS.batch_size,
    #     num_threads=FLAGS.batch_thread_number,
    #     capacity=16,
    #     min_after_dequeue=FLAGS.min_after_dequeue)
    batch_features = tf.train.batch(
        examples,
        batch_size=FLAGS.batch_size,
        dynamic_pad=True)
    item2vec = tf.nn.embedding_lookup_sparse(aa, batch_features['education_num'], None, combiner="sum")

    wide_features = tf.feature_column.input_layer(batch_features, wide_columns)
    label = tf.feature_column.input_layer(batch_features, label_columns)
    deep_features = tf.concat([tf.feature_column.input_layer(batch_features, deep_columns),
                               item2vec],1)


    return label, deep_features



def read_and_decode_tfrecords2(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    examples = tf.parse_single_example(
        serialized_example,
        features={
            "education_num": tf.FixedLenFeature([], tf.int64),
            "age": tf.FixedLenFeature([], tf.int64),
        })
    label = examples["education_num"]
    features = examples["age"]
    return label, features


def read_and_decode_csv(filename_queue):
    # Notice that it supports label in the last column only
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[1.0] for i in range(FLAGS.feature_size)] + [[0]]
    columns = tf.decode_csv(value, record_defaults=record_defaults)
    label = columns[-1]
    features = tf.stack(columns[0:-1])
    return label, features


def full_connect(inputs, weights_shape, biases_shape, is_train=True):
    weights = tf.get_variable(
        "weights", weights_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable(
        "biases", biases_shape, initializer=tf.random_normal_initializer())
    layer = tf.matmul(inputs, weights) + biases

    if FLAGS.enable_bn and is_train:
        mean, var = tf.nn.moments(layer, axes=[0])
        scale = tf.get_variable(
            "scale", biases_shape, initializer=tf.random_normal_initializer())
        shift = tf.get_variable(
            "shift", biases_shape, initializer=tf.random_normal_initializer())
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                          FLAGS.bn_epsilon)
    return layer


def full_connect_relu(inputs, weights_shape, biases_shape, is_train=True):
    layer = full_connect(inputs, weights_shape, biases_shape, is_train)
    layer = tf.nn.relu(layer)
    return layer


def customized_inference(inputs, input_units, output_units, is_train=True):
    hidden1_units = 128
    hidden2_units = 32
    hidden3_units = 8

    with tf.variable_scope("input"):
        layer = full_connect_relu(inputs, [input_units, hidden1_units],
                                  [hidden1_units], is_train)
    with tf.variable_scope("layer0"):
        layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                                  [hidden2_units], is_train)
    with tf.variable_scope("layer1"):
        layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                                  [hidden3_units], is_train)
    if FLAGS.enable_dropout and is_train:
        layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)
    with tf.variable_scope("output"):
        layer = full_connect(layer, [hidden3_units, output_units], [output_units],
                             is_train)
    return layer


logging.basicConfig(level=logging.INFO)
FLAGS = define_flags()
assert_flags(FLAGS)
pprint.PrettyPrinter().pprint(FLAGS.__flags)
if FLAGS.enable_colored_log:
    import coloredlogs

    coloredlogs.install()


def main():
    # Get hyper-parameters
    if os.path.exists(FLAGS.checkpoint_path) == False:
        os.makedirs(FLAGS.checkpoint_path)

    if os.path.exists(FLAGS.output_path) == False:
        os.makedirs(FLAGS.output_path)

    EPOCH_NUMBER = FLAGS.epoch_number

    BATCH_CAPACITY = FLAGS.batch_thread_number * FLAGS.batch_size + FLAGS.min_after_dequeue

    if FLAGS.train_file_format == "tfrecords":
        read_and_decode_function = read_and_decode_tfrecords
    elif FLAGS.train_file_format == "csv":
        read_and_decode_function = read_and_decode_csv

    train_filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(FLAGS.train_file), num_epochs=EPOCH_NUMBER)
    train_label, train_features = read_and_decode_function(train_filename_queue)

    # batch_labels, batch_features = tf.train.shuffle_batch(
    #     [train_label, train_features],
    #     batch_size=FLAGS.batch_size,
    #     num_threads=FLAGS.batch_thread_number,
    #     capacity=BATCH_CAPACITY,
    #     min_after_dequeue=FLAGS.min_after_dequeue,
    # )
    # batch_labels, batch_features = tf.train.batch(
    #     [train_label, train_features],
    #     batch_size=FLAGS.batch_size,
    #     num_threads=FLAGS.batch_thread_number,
    #     capacity=BATCH_CAPACITY,
    #
    # )

    logging.info("Use the model: {}, model network: {}".format(
        FLAGS.model, FLAGS.dnn_struct))

    init_op = [
        tf.global_variables_initializer(),
        tf.local_variables_initializer(),
        tf.tables_initializer()
    ]

    # Create session to run
    with tf.Session() as sess:
        sess.run(init_op)

        if FLAGS.mode == "train":

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            try:
                while not coord.should_stop():
                    if FLAGS.enable_benchmark:
                        sess.run(train_features)
                    else:
                        feature_print = sess.run([train_label])
                        print('----')
                        print(np.transpose(feature_print))
            except tf.errors.OutOfRangeError:
                if FLAGS.enable_benchmark:
                    print("Finish training for benchmark")
                    exit(0)
                else:
                    # Export the model after training
                    print("Do not export the model yet")

            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    main()
