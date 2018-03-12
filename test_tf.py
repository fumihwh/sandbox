from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from onnx_tf.backend import run_model, run_node
import onnx
import itertools

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

# node = onnx.helper.make_node(
#   'AveragePool',
#   inputs=['x'],
#   outputs=['y'],
#   kernel_shape=[5, 5],
#   strides=[3, 3]
# )
# x = np.random.randn(1, 3, 32, 32).astype(np.float32)
# x_shape = np.shape(x)
# kernel_shape = (5, 5)
# strides = (3, 3)
# out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
# padded = x
# y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0))
# sess = tf.InteractiveSession()
# x = np.transpose(x, (0, 2, 3, 1))
# # x = np.transpose(x, (0, 2, 1))
# tensor_x = tf.convert_to_tensor(x, np.float32)
# rs = tf.nn.avg_pool(tensor_x, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID')
# _rs = sess.run(rs)
# print(_rs)

# node = onnx.helper.make_node(
#   'AveragePool',
#   inputs=['x'],
#   outputs=['y'],
#   kernel_shape=[5, 5],
#   strides=[2, 2],
# )
# x = np.array([[[
#   [1, 2, 3, 4, 5],
#   [6, 7, 8, 9, 10],
#   [11, 12, 13, 14, 15],
#   [16, 17, 18, 19, 20],
#   [21, 22, 23, 24, 25],
# ]]]).astype(np.float32)
# x_shape = (1, 1, 5, 5)
# kernel_shape = (5, 5)
# strides = (1, 1)
# # out_shape = (3, 3)
# out_shape = get_output_shape('VALID', [9, 9], kernel_shape, strides)
# padded = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant', constant_values=np.nan)
# y = pool(padded, x_shape, kernel_shape, strides, out_shape, (4, 4))

# x_shape = (1, 1, 5, 5)
# kernel_shape = (2, 2)
# strides = (2, 2)
# out_shape = (4, 4)
# padded = x
# y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0))
# print(y)


node = onnx.helper.make_node(
  'AveragePool',
  inputs=['x'],
  outputs=['y'],
  kernel_shape=[3, 3],
  pads=[2, 2, 2, 2]
)
x = np.random.randn(1, 3, 28, 28).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (3, 3)
strides = (1, 1)
pad_bottom = 2
pad_top = 2
pad_right = 2
pad_left = 2
pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=np.nan)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape)

sess = tf.InteractiveSession()
# x = np.random.randn(1, 32, 32, 3).astype(np.float32)
x = np.transpose(x, (0, 2, 3, 1))
# x = np.transpose(x, (0, 2, 1))
tensor_x = tf.convert_to_tensor(x, np.float32)
# l2 = tf.norm(tensor_x)
# rs = tf.depth_to_space(tensor_x, block_size=2)
# mean, variance = tf.nn.moments(x, [0])
# tf.nn.batch_normalization(x, mean, variance, None, None, 1e-5)
rs = tf.nn.avg_pool(tensor_x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
# rs = tf.nn.pool(tensor_x, window_shape=[2], strides=[1], padding='VALID', pooling_type='AVG')
# # v_rs = tf.nn.avg_pool(tensor_x, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding='VALID')
# tf_rs = sess.run(rs)
# # tf_v_rs = sess.run(v_rs)
#
# # print(np.shape(np_rs))
# # print(np.shape(tf_rs))
# # print(np.shape(tf_v_rs))
# #
# # print(np.allclose(np.transpose(y, (0, 2, 3, 1)), tf_v_rs, rtol=1e-3))
# print(np.allclose(np.transpose(y, (0, 2, 1)), tf_rs, rtol=1e-3))

# a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
# b = tf.constant([[100], [101],], name='b')
# add_op = a + b
#
_rs = sess.run(rs)
print(np.transpose(_rs, (0, 3, 1, 2)))
