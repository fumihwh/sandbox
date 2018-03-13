import os

import numpy as np
import onnx
from onnx import helper
from caffe2.python.onnx import backend
import tensorflow as tf
from tensorflow.python.platform import gfile
from onnx_tf.frontend import convert_graph
from onnx_tf.backend import run_model

import numpy as np

def test():
  input = np.random.randn(1, 3, 32, 32)

  model = onnx.load('pb/onnx_lmnet_cifar10_nchw.pb')
  # outputs_tf = run_model(model, [input])
  node_def = helper.make_node("Reshape", ["X"], ["Y"], shape=[10, 10])
  x = np.random.randn(100)
  output = backend.run_node(node_def, [x])
  print(output)

  outputs_caffe2 = backend.run_model(model, [input])

  print(outputs_caffe2)
  # print(outputs_tf)

  print(np.allclose(outputs_caffe2, outputs_tf, rtol=1e-3))

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
