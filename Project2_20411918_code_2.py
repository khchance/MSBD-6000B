from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd
import tables

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])

  # Convolutional Layer #1
  # Computes 96 features using a 7x7 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 224, 224, 3]
  # Output Tensor Shape: [batch_size, 56, 56, 96]
  conv1 = tf.layers.conv2d(inputs=input_layer, filters=96, kernel_size=[11, 11], strides=4, padding="same", activation=tf.nn.relu, kernel_initializer=None)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 56, 56, 96]
  # Output Tensor Shape: [batch_size, 28, 28, 96]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 256 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 96]
  # Output Tensor Shape: [batch_size, 28, 28, 256]
  conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu, kernel_initializer=None)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 256]
  # Output Tensor Shape: [batch_size, 14, 14, 256]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3
  # Computes 256 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 256]
  # Output Tensor Shape: [batch_size, 14, 14, 256]
  conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu, kernel_initializer=None)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 256]
  # Output Tensor Shape: [batch_size, 7, 7, 256]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 256]
  # Output Tensor Shape: [batch_size, 7 * 7 * 256]
  pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 256])

  # Dense Layer #1
  # Densely connected layer with 512 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 256]
  # Output Tensor Shape: [batch_size, 512]
  dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer #2
  # Densely connected layer with 512 neurons
  # Input Tensor Shape: [batch_size, 512]
  # Output Tensor Shape: [batch_size, 512]
  dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 512]
  # Output Tensor Shape: [batch_size, 5]
  logits = tf.layers.dense(inputs=dropout2, units=5)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training, eval and test data
  hdf5_path = 'dataset_eq.hdf5'
  hdf5_file = tables.open_file(hdf5_path, mode='r')

  train_data = np.asarray(hdf5_file.root.train_img)
  train_labels = np.asarray(hdf5_file.root.train_labels)

  eval_data = np.asarray(hdf5_file.root.val_img)
  eval_labels = np.asarray(hdf5_file.root.val_labels)

  test_data = np.asarray(hdf5_file.root.test_img)

  # Create the Estimator
  image_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="./model")
  
  # Prepare validation data
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
  # Prepare test data
  test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, num_epochs=1, shuffle=False)

  max_acc = 0
  patient = 0
  for i in range(2000):
      # Train the model
      train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
      image_classifier.train(input_fn=train_input_fn, steps=1)

      # Monitor the validation accuarcy every 20 iterations
      if i % 20 == 0:
          eval_results = image_classifier.evaluate(input_fn=eval_input_fn)
          print(eval_results)
          # Store the best model and the corresponding prediction
          if eval_results['accuracy'] > max_acc:
              max_acc = eval_results['accuracy']
              test_results = image_classifier.predict(input_fn=test_input_fn)
              pred = {'classes':[], 'probabilities':[]}
              for j in test_results:
                  pred['probabilities'].append(np.max(j['probabilities']))
                  pred['classes'].append(j['classes'])

              pred = pd.DataFrame(pred)
              pred.to_csv('./model/Predict_%f.csv' %eval_results['accuracy'], index=False)
              patient = 0
      patient += 1

      # Implement early stopping if validation accuracy does not improve for 100 iterations
      if patient > 100: break


if __name__ == "__main__":
  tf.app.run()
