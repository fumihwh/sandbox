import os

import onnx
import onnx_caffe2.backend
import tensorflow as tf
from tensorflow.python.platform import gfile
from onnx_tf.frontend import convert_graph
from onnx_tf.backend import run_model

import numpy as np

def test():
  input = np.random.randn(10, 784)

  model = onnx.load('pb/onnx_gh_mnist.pb')
  outputs = onnx_caffe2.backend.run_model(model, [input])

  print(outputs)

  # tf_graph = 'tf_gh_minst.pb'
  #
  # with tf.Session() as sess:
  #   model_filename = os.path.join('pb', tf_graph)
  #   with gfile.FastGFile(model_filename, 'rb') as f:
  #     graph_def = tf.GraphDef()
  #     graph_def.ParseFromString(f.read())
  #   with sess.graph.as_default() as graph:
  #     tf.import_graph_def(
  #       graph_def,
  #       name=''
  #     )
  #   saver = tf.train.import_meta_graph(os.path.join('pb', tf_graph))
  #   saver.restore(sess)
  #   output = sess.graph.get_tensor_by_name('output:0')
  #   data_placeholder = tf.get_collection('input_ph')[0]
  #   rs_tf = sess.run(output, feed_dict={sess.graph.get_tensor_by_name('input_ph:0'): input})
  #   print(rs_tf)

test()
