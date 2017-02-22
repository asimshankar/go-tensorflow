# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# stipped down to just the creation of the graph and the addition of a saver
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

def create_graph(vocabulary_size, embedding_size, num_sampled, filename):
    """Create the graph for training a word2vec embedding.

    Write the GraphDef to 'filename' and prints out the names of important
    nodes in the graph.

    This function is essentially the graph creation portion of:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    """
    with tf.Graph().as_default():
      # Input data.
      train_examples = tf.placeholder(tf.int32, shape=[None], name='train_examples')
      train_labels = tf.placeholder(tf.int32, shape=[None, 1], name='train_labels')
      valid_dataset = tf.placeholder(tf.int32, shape=[None], name='input')
      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
          name='embedding')
      embed = tf.nn.embedding_lookup(embeddings, train_examples)
    
      # Construct the variables for the NCE loss
      nce_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)),
          name='nce_weights')
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='nce_biases')
      # END CPU
    
      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size),
          name='loss')
    
      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss, name='train_op')
    
      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = tf.divide(embeddings, norm, name='normalized_embeddings')
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True, name='similarity')
    
      # Add variable initializer.
      init = tf.global_variables_initializer()

      with open(filename, 'w') as f:
          f.write(tf.get_default_graph().as_graph_def().SerializeToString())

      print("""
Saved GraphDef to: {filename}

TRAINING:
(use this in main.go):

func newTrainer(graph *tf.Graph, sess *tf.Session) (*trainer, error) {{
    t := &trainer{{
      sess:     sess,
      examples: graph.Operation("{examples}").Output(0),
      labels:   graph.Operation("{labels}").Output(0),
      train:    graph.Operation("{train_op}"),
      loss:     graph.Operation("{loss}").Output(0),
    }}
    // Initialize the variables
    _, err := sess.Run(nil, nil, []*tf.Operation{{graph.Operation("{init}")}})
    return t, err
}}

VALIDATION/INFERENCE:
Input placeholder:     {input}
Normalized Embeddings: {normalized_embeddings}
Cosine similarity:     {similarity}
""".format(
    filename=filename, 
    init=init.name,
    examples=train_examples.op.name,
    labels=train_labels.op.name,
    train_op=optimizer.name,
    loss=loss.op.name,
    input=valid_dataset.op.name,
    normalized_embeddings=normalized_embeddings.op.name,
    similarity=similarity.op.name))

if __name__ == '__main__':
  create_graph(50000, 128, 64, 'word2vec_train_graph.pb')

