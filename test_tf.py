from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from onnx_tf.backend import run_node
from onnx import helper
import itertools
import tensorflow as tf

def get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, output_spatial_shape):
  pad_shape = [0] * len(input_spatial_shape)
  if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
    for i in range(len(input_spatial_shape)):
      pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - \
                     input_spatial_shape[i]
  elif auto_pad == 'VALID':
    pass
  return pad_shape

def get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial):
  out_shape = [0] * len(input_spatial_shape)
  if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
    for i in range(len(input_spatial_shape)):
      out_shape[i] = int(np.ceil(float(input_spatial_shape[i]) / float(strides_spatial[i])))
  elif auto_pad == 'VALID':
    for i in range(len(input_spatial_shape)):
      out_shape[i] = int(
        np.ceil(float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
  return out_shape

def pool(padded, x_shape, kernel_shape, strides_shape, out_shape, pad_shape):
  spatial_size = len(x_shape) - 2
  y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

  for shape in itertools.product(range(x_shape[0]),
                                 range(x_shape[1]),
                                 *[range(
                                   int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i] + 1))
                                   for i in range(spatial_size)]):
    window = padded[shape[0], shape[1]]
    window_vals = np.array([window[i] for i in list(
      itertools.product(
        *[range(strides_shape[i] * shape[i + 2], strides_shape[i] * shape[i + 2] + kernel_shape[i]) for i in
          range(spatial_size)])
    )])
    average = np.average(window_vals[np.where(~np.isnan(window_vals))])
    y[shape] = average
  return y.astype(np.float32)

# with tf.Graph().as_default() as graph:
#   # x = tf.Variable(np.random.randint(10, size=(1, 5, 5, 3)), dtype=tf.float32)
#   input = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5, 3), name='ph')
#   x = tf.layers.conv2d(input,
#                        filters=16,
#                        kernel_size=3,
#                        padding='SAME',
#                        use_bias=False,
#                        name='conv_ly')
#   bias = tf.get_variable('bias', shape=(16,), initializer=tf.random_uniform_initializer)
#   x = tf.nn.bias_add(x, bias, data_format='NHWC', name='biasadd')
#   output = x * bias
#   output = tf.identity(output, name='output')
#
#   sess = tf.Session(graph=graph)
#   sess.run(tf.global_variables_initializer())
#   _rs = sess.run(x, feed_dict={input: np.random.randn(1, 5, 5, 3)})
#   print(_rs)
#   minimal_graph = tf.graph_util.convert_variables_to_constants(
#     sess,
#     sess.graph.as_graph_def(add_shapes=True),
#     ['output'],
#   )
#   tf.train.write_graph(minimal_graph, 'pb', 'tf_conv_new.pb', as_text=False)

# node_def = helper.make_node("Relu", ["X"], ["Y"])
# output = run_node(node_def, [[-0.1, 0.1]])
# print(output["Y"])
# import onnx
#
# from onnx import checker, helper, TensorProto
#
# conv = helper.make_node("Conv", ["X", "Y"], ["Z"])
# add = helper.make_node("Add", ["Z", "A"], ["B"], broadcast=1, axis=1)
# graph = helper.make_graph(
#   [conv, add],
#   "test",
#   [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3)),
#    helper.make_tensor_value_info("Y", TensorProto.FLOAT, (16, 5, 3, 3)),
#    helper.make_tensor_value_info("A", TensorProto.FLOAT, (16,))],
#   [helper.make_tensor_value_info("B", TensorProto.FLOAT, (1, 16, 1, 1))],
#   value_info=[
#     helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 16, 1, 1)),
#   ]
# )
#
# model = helper.make_model(graph)
# onnx.save(model, 'pb/onnx_conv_1d.pb')

with tf.Graph().as_default() as graph:
  # x = tf.Variable(np.random.randint(10, size=(1, 5, 5, 3)), dtype=tf.float32)
  input = tf.placeholder(dtype=tf.float32, shape=(1, 128, 128, 3), name='ph')
  x = tf.image.resize_nearest_neighbor(input, (100, 100))
  output = tf.identity(x, name='output')
  sess = tf.Session(graph=graph)
  sess.run(tf.global_variables_initializer())
  _rs = sess.run(x, feed_dict={input: np.random.randn(1, 128, 128, 3)})
  print(_rs)
  minimal_graph = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(add_shapes=True),
    ['output'],
  )
  tf.train.write_graph(minimal_graph, 'pb', 'tf_resize.pb', as_text=False)
