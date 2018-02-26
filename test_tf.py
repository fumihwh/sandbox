from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from onnx_tf.backend import run_model, run_node
import onnx


def hardmax_2d(x):
    return np.eye(x.shape[1])[np.argmax(x, axis=1)]


node = onnx.helper.make_node(
    'Pow',
    inputs=['x', 'y'],
    outputs=['z'],
    broadcast=1,
    axis=0,
)
# x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
# y = np.array([2, 3, 4]).astype(np.float32)
# y = np.array([[2, 3, 4]]).astype(np.float32)
# z = np.array([[1, 8, 81], [16, 125, 1296]]).astype(np.float32)

# x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
# y = np.array([2, 3]).astype(np.float32)
# z = np.array([[1, 4, 9], [64, 125, 216]]).astype(np.float32)
#
# z1 = run_node(node, [x, y])
# print(x)
# print(y)
# print(z)
# print(z1)

# print(np.allclose(y, z, rtol=1e-3))
# print(np.allclose(z, z1, rtol=1e-3))

x = np.random.rand(20, 1, 50, 16)
sess = tf.InteractiveSession()
tensor_x = tf.convert_to_tensor(x, np.float32)
rs = tf.nn.max_pool(tensor_x, ksize=[1, 1, 3, 1], strides=[1, 1, 2, 1], padding='VALID')
print(sess.run(rs))
print(np.shape(sess.run(rs)))
# tensor_y = tf.convert_to_tensor(y, np.float32)
# z = tf.broadcast_static_shape(tensor_x.get_shape(),
#                               tensor_y.get_shape())
# z1 = tf.broadcast_dynamic_shape(tensor_x.get_shape(),
#                                 tensor_y.get_shape())
# print(z)
# print(z1)
# print(tf.reshape(tensor_y, z1).eval())
#
