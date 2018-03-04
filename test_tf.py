from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import torch
from onnx_tf.backend import run_model, run_node
import onnx
import itertools


def _get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, output_spatial_shape):
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - \
                           input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape


def _get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial):
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(input_spatial_shape[i] / strides_spatial[i]))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil((input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / strides_spatial[i]))
    return out_shape


def _pool_2d(padded, x_shape, out_shape, kernel_shape, strides, pad_shape):
    k_h, k_w = kernel_shape
    s_h, s_w = strides
    y = np.zeros((x_shape[0], x_shape[1], out_shape[0], out_shape[1]))
    for n, c, h, w in itertools.product(range(x_shape[0]),
                                        range(x_shape[1]),
                                        range(int((x_shape[2] + pad_shape[0] - k_h) / s_h + 1)),
                                        range(int((x_shape[3] + pad_shape[1] - k_w) / s_w + 1))):
        window = padded[n, c, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w]
        average = np.average(window[np.where(~np.isnan(window))])
        y[n, c, h, w] = average
    return y


def _pool(padded, spatial_size, kernel_shape, strides_shape, out_shape, pad_shape):
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(range(x_shape[0]),
                                   range(x_shape[1]),
                                   *[range(
                                       int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i] + 1))
                                       for i in range(spatial_size)]):
        window = padded[shape[0], shape[1]]
        if spatial_size == 1:
            window = window[
                     strides_shape[0] * shape[2]:strides_shape[0] * shape[2] + kernel_shape[0]]
        elif spatial_size == 2:
            window = window[
                     strides_shape[0] * shape[2]:strides_shape[0] * shape[2] + kernel_shape[0],
                     strides_shape[1] * shape[3]:strides_shape[1] * shape[3] + kernel_shape[1]]
        elif spatial_size == 3:
            window = window[
                     strides_shape[0] * shape[2]:strides_shape[0] * shape[2] + kernel_shape[0],
                     strides_shape[1] * shape[3]:strides_shape[1] * shape[3] + kernel_shape[1],
                     strides_shape[2] * shape[4]:strides_shape[2] * shape[4] + kernel_shape[2]]
        average = np.average(window[np.where(~np.isnan(window))])
        if spatial_size == 1:
            y[shape[0], shape[1], shape[2]] = average
        elif spatial_size == 2:
            y[shape[0], shape[1], shape[2], shape[3]] = average
        elif spatial_size == 3:
            y[shape[0], shape[1], shape[2], shape[3], shape[4]] = average
    return y.astype(np.float32)


# node = onnx.helper.make_node(
#   'AveragePool',
#   inputs=['x'],
#   outputs=['y'],
#   kernel_shape=[2],
# )
# x = np.random.randn(1, 3, 32).astype(np.float32)
# x_shape = np.shape(x)
# spatial_size = np.ndim(x) - 2
# kernel_shape = [2]
# strides = [1]
# out_shape = _get_output_shape('VALID', x_shape[2:3], kernel_shape, strides)
# padded = x
# y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, [0])

# print(y)

# x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).astype(np.float32)
# x = np.random.randn(32, 32).astype(np.float32)
x = [[[[1, 2, 3, 4],
       [5, 6, 7, 8]],
      [[9, 10, 11, 12],
       [13, 14, 15, 16]]]]
sess = tf.InteractiveSession()
# x = np.random.randn(1, 32, 32, 3).astype(np.float32)
# # x = np.transpose(x, (0, 2, 3, 1))
# # x = np.transpose(x, (0, 2, 1))
tensor_x = tf.convert_to_tensor(x, np.float32)
# l2 = tf.norm(tensor_x)
rs = tf.depth_to_space(tensor_x, block_size=2)
# mean, variance = tf.nn.moments(x, [0])
# tf.nn.batch_normalization(x, mean, variance, None, None, 1e-5)
# # rs = tf.nn.avg_pool(tensor_x, ksize=[1, 2, 1], strides=[1, 1, 1], padding='SAME')
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
print(_rs)
