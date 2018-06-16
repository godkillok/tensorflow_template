import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
# We know that MNIST images are 28 pixels in each dimension.1
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

num_epochs=10

batch_size=100

def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn,
    #           see e.g. train_input_fn for these two.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.


    x = features
    print(x.shape)
    # The convolutional layers expect 4-rank tensors
    # but x is a 2-rank tensor, so reshape it.
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=16, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=36, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Flatten to a 2-rank tensor.
    net = tf.contrib.layers.flatten(net)
    # Eventually this should be replaced with:
    # net = tf.layers.flatten(net)

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=10)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)



    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.

        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }


        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec


data = input_data.read_data_sets('./data/MNIST/', one_hot=True)
params = {"learning_rate": 1e-4}

model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./checkpoints_tutorial17-2/")


train_path='./data/MNIST/'
import os
import sys

def search(train_path, word):
    file=[]
    for filename in os.listdir(train_path):
        fp = os.path.join(train_path, filename)
        if os.path.isfile(fp) and word in filename and '.tfrecords':
            file.append(fp)
    return file


def train_input_fn():
    '''
    训练输入函数，返回一个 batch 的 features 和 labels
    '''

    filenames = search(train_path,word='train')

    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.map(parser)
    train_dataset=train_dataset.shuffle(buffer_size=100).batch(batch_size).repeat(10)
    # train_dataset = train_dataset.shuffle(train_dataset,100)
    # # num_epochs 为整个数据集的迭代次数
    # train_dataset = train_dataset.batch(batch_size)
    #
    # train_dataset = train_dataset.repeat(num_epochs)

    train_iterator = train_dataset.make_one_shot_iterator()

    features, labels = train_iterator.get_next()
    return features, labels

def test_input_fn():
    '''
    训练输入函数，返回一个 batch 的 features 和 labels
    '''

    filenames = search(train_path,word='test')

    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.map(parser)
    train_dataset = train_dataset.batch(batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()
    features, labels = train_iterator.get_next()

    return features, labels

def predict_input_fn():
    '''
    训练输入函数，返回一个 batch 的 features 和 labels
    '''

    filenames = search(train_path,word='test')

    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.map(parser)

    # train_dataset = train_dataset.shuffle(train_dataset,100)
    # num_epochs 为整个数据集的迭代次数

    train_dataset = train_dataset.batch(batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()

    features,_ = train_iterator.get_next()
    return features

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


dataset,gd=train_input_fn()

count=0
with tf.Session() as sess:
    try:
        while True:
            sess.run(gd)
            count += 1
            print(count)
    except tf.errors.OutOfRangeError:
        print("end!")

import time
start=time.time()
test=model.train(input_fn=train_input_fn,steps=3)
print(time.time()-start)



start=time.time()
result = model.evaluate(input_fn=test_input_fn)
print(result)
print(time.time()-start)

predictions = model.predict(input_fn=predict_input_fn)
count=0
for p in predictions:
    count+=1

print(count)
cls = [p for p in predictions]


