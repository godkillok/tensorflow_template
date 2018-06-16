import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image

#把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def genmnsit_tfreords(words='train'):
    #读取MNIST数据
    mnist = input_data.read_data_sets("./", dtype=tf.uint8, one_hot=True)

    if words=='train':
        #训练数据的图像，可以作为一个属性来存储
        images = mnist.train.images
        #训练数据所对应的正确答案，可以作为一个属性来存储
        labels = mnist.train.labels
        num_examples = mnist.train.num_examples
    else:
        images = mnist.test.images
        labels = mnist.test.labels
        num_examples = mnist.test.num_examples

    #训练数据的图像分辨率，可以作为一个属性来存储
    pixels = images.shape[0]
    #训练数据的个数

    #指定要写入TFRecord文件的地址

    filename = "./mnist_{}_0.tfrecords".format(words)
    # 创建一个write来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        #把图像矩阵转化为字符串
        image_raw = images[index].tostring()
        if index%10000==0:
            writer.close()
            filename = "./mnist_{}_{}.tfrecords".format(words,int(index/10000))
            # 创建一个write来写TFRecord文件
            writer = tf.python_io.TFRecordWriter(filename)

        #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            #'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        #将 Example 写入TFRecord文件
        writer.write(example.SerializeToString())

    writer.close()
mnist = input_data.read_data_sets("./", dtype=tf.uint8, one_hot=True)
genmnsit_tfreords()
# genmnsit_tfreords('test')