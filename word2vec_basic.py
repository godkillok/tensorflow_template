
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

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
import datetime


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


data_index = 0

batch_size = 1024
embedding_size = 32  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)



EPOCH_NUMBER=20
directory =[]

for i in range(1, 555576, 10000):
    output_filename="C:/work/tensorflow_template/tags_{}.tfrecords".format(i)
    directory.append(output_filename)

train_filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(directory), num_epochs=EPOCH_NUMBER)

def read_and_decode_tfrecords(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  examples = tf.parse_single_example(
      serialized_example,
      features={
          "label": tf.FixedLenFeature([1], tf.int64),
          "values": tf.FixedLenFeature([], tf.int64),
      })
  label = examples["label"]
  features = examples["values"]
  return label, features

train_label, train_features = read_and_decode_tfrecords(train_filename_queue)
batch_thread_number=4
min_after_dequeue=512
BATCH_CAPACITY = batch_thread_number * batch_size + min_after_dequeue

batch_labels, batch_features = tf.train.shuffle_batch(
  [train_label, train_features],
  batch_size=batch_size,
  num_threads=batch_thread_number,
  capacity=BATCH_CAPACITY,
  min_after_dequeue=min_after_dequeue)

import  codecs
reverse_dictionary={}
with codecs.open('./metadata.tsv', 'r','utf8') as f:
  all_line_txt = f.readlines()
  for (i,l) in enumerate(all_line_txt):
    reverse_dictionary[i]=l

vocabulary_size=i+1

with tf.name_scope('embeddings'):
  embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, batch_features)

# Construct the variables for the NCE loss
with tf.name_scope('weights'):
  nce_weights = tf.Variable(
      tf.truncated_normal(
          [vocabulary_size, embedding_size],
          stddev=1.0 / math.sqrt(embedding_size)))
with tf.name_scope('biases'):
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
# Explanation of the meaning of NCE loss:
#   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
with tf.name_scope('loss'):
  loss = tf.reduce_mean(
      tf.nn.nce_loss(
          weights=nce_weights,
          biases=nce_biases,
          labels=batch_labels,
          inputs=embed,
          num_sampled=num_sampled,
          num_classes=vocabulary_size))

# Add the loss value as a scalar to summary.
tf.summary.scalar('loss', loss)


starter_learning_rate=0.3
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    100000,
    FLAGS.lr_decay_rate,
    staircase=True)


# Construct the SGD optimizer using a learning rate of 1.0.
with tf.name_scope('optimizer'):
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm


# Merge all summaries.
merged = tf.summary.merge_all()

# Add variable initializer.
init = (tf.global_variables_initializer(), tf.local_variables_initializer())

# Create a saver.
saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 1000

with tf.Session( ) as session:
  # Open a writer to write summaries.
  writer = tf.summary.FileWriter(FLAGS.log_dir)

  # We must initialize all variables before we use them.
  session.run(init)
  print('Initialized')

  average_loss = 0

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=session)
  start_time = datetime.datetime.now()
  step=0
  try:
    while not coord.should_stop():
      step+=1
      run_metadata = tf.RunMetadata()

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
      # Feed metadata variable to session for visualizing the graph in TensorBoard.
      _, summary, loss_val = session.run(
          [optimizer, merged, loss],
          run_metadata=run_metadata)
      average_loss += loss_val

      # Add returned summaries to writer in each step.
      writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)

      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0
  except tf.errors.OutOfRangeError:
    writer.add_run_metadata(run_metadata, 'step%d' % step)
    print('done!')
  final_embeddings = normalized_embeddings.eval()

  # Write corresponding labels for the embeddings.

  # Save the model for checkpoints.
  saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))
  endtime=datetime.datetime.now()
  print('spend time {} minutes'.format(endtime-start_time))
  # Create a configuration for visualizing embeddings with the labels in TensorBoard.
  config = projector.ProjectorConfig()
  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = embeddings.name
  embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
  projector.visualize_embeddings(writer, config)

writer.close()

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)