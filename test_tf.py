from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from onnx_tf.backend import run_node
from onnx import helper
import onnx
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
# constant_fill = helper.make_node("ConstantFill", ["X"], ["Z"])
# graph = helper.make_graph(
#   [constant_fill],
#   "test",
#   [helper.make_tensor_value_info("X", TensorProto.FLOAT, (1, 5, 3, 3))],
#   [helper.make_tensor_value_info("Z", TensorProto.FLOAT, (1, 5, 3, 3))],
# )
#
# model = helper.make_model(graph)
# onnx.save(model, 'pb/onnx_constant_fill.pb')

#
# with tf.Graph().as_default() as graph:
#   # x = tf.Variable(np.random.randint(10, size=(1, 5, 5, 3)), dtype=tf.float32)
#   x = tf.placeholder(dtype=tf.float32, shape=(1, 2, 3, 4))
#   y = tf.tile(x, (1, 2, 3, 4))
#   # x = tf.image.resize_nearest_neighbor(input, (100, 100))
#   output = tf.identity(y, name='output')
#
#   sess = tf.Session(graph=graph)
#   sess.run(tf.global_variables_initializer())
#   # _rs = sess.run(x, feed_dict={input: np.random.randn(1, 128, 128, 3)})
#   _rs = sess.run(output, feed_dict={x: np.random.randn(1, 2, 3, 4)})
#   print(np.size(np.random.randn(1, 2, 3, 4)))
#   print(_rs)
#   minimal_graph = tf.graph_util.convert_variables_to_constants(
#     sess,
#     sess.graph.as_graph_def(add_shapes=True),
#     ['output'],
#   )
#   tf.train.write_graph(minimal_graph, 'pb', 'tf_tile.pb', as_text=False)

# model = onnx.load('/Users/wenhao/Projects/onnx/onnx/backend/test/data/pytorch-operator/test_operator_lstm/model.onnx')
# print(model)

import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg

from onnx_tf.frontend import convert_graph, tensorflow_graph_to_onnx_model


def get_node_by_name(nodes, name):
    for node in nodes:
        if node.name == name:
            return node
    raise ValueError("Node {} is not found in the graph provided".format(name))


# Convert TF slim VGG 16 to ONNX
# with tf.Session() as sess:
#     # Setup input placeholder
#     batch_size = 1
#     height, width = vgg.vgg_16.default_image_size, vgg.vgg_16.default_image_size
#     inputs = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3))
#
#     # Build model
#     predictions, model = vgg.vgg_16(inputs)
#
#     # Run the initializer
#     sess.run(tf.global_variables_initializer())
#
#     # Get the graph definition and preprocess it
#     # tfgraphdef = tf.get_default_graph().as_graph_def(add_shapes=True)
#     # tfgraphdef = tf.graph_util.remove_training_nodes(tfgraphdef)
#     # tfgraphdef = tf.graph_util.convert_variables_to_constants(sess, tfgraphdef, [predictions.op.name])
#     tfgraphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [predictions.op.name])
#
#     # Get output NodeDef
#     output_node_def = get_node_by_name(tfgraphdef.node, 'vgg_16/fc8/squeezed')
#
#     # Convert default TF graph to ONNX graph
#     # onnx_graph = convert_graph(tfgraphdef, output_node_def)
#     onnx_graph = tensorflow_graph_to_onnx_model(tfgraphdef, 'vgg_16/fc8/squeezed', ignore_unimplemented=True)

# tf.InteractiveSession()
# np_x = np.array([[[[0, 1],
#                    [2, 3],
#                    [4, 5],
#                    [6, 7]],
#                   [[8, 9],
#                    [10, 11],
#                    [12, 13],
#                    [14, 15]],
#                   [[16, 17],
#                    [18, 19],
#                    [20, 21],
#                    [22, 23]],
#                   [[24, 25],
#                    [26, 27],
#                    [28, 29],
#                    [30, 31]]],
#                  [[[32, 33],
#                    [34, 35],
#                    [36, 37],
#                    [38, 39]],
#                   [[40, 41],
#                    [42, 43],
#                    [44, 45],
#                    [46, 47]],
#                   [[48, 49],
#                    [50, 51],
#                    [52, 53],
#                    [54, 55]],
#                   [[56, 57],
#                    [58, 59],
#                    [60, 61],
#                    [62, 63]]]]).astype(np.float32)
# x = tf.convert_to_tensor(np_x, np.float32)
# rs = tf.nn.max_pool_with_argmax(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
# print(repr(rs[0].eval()))
# print(repr(rs[1].eval()))

# inputs = np.array([[[[10., 11.],
#                      [14., 15.]],
#
#                     [[26., 27.],
#                      [30., 31.]]],
#
#                    [[[42., 43.],
#                      [46., 47.]],
#
#                     [[58., 59.],
#                      [62., 63.]]]])
#
# mask = np.array([[[[10, 11],
#                    [14, 15]],
#
#                   [[26, 27],
#                    [30, 31]]],
#
#                  [[[10, 11],
#                    [14, 15]],
#
#                   [[26, 27],
#                    [30, 31]]]])
# input_shape = inputs.shape
# ksize = (1, 2, 2, 1)
# #  calculation new shape
# output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
# # calculation indices for batch, height, width and feature maps
# one_like_mask = np.ones_like(mask)
# batch_range = np.reshape(list(range(output_shape[0])), [input_shape[0], 1, 1, 1])
# # batch_range = tf.Print(
# #     batch_range,
# #     [batch_range],
# #     message="batch_range:",
# #     summarize=2000,
# # )
# b = one_like_mask * batch_range
# y = mask // (output_shape[2] * output_shape[3])
# x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
# channel_range = list(range(output_shape[3]))
# c = one_like_mask * channel_range
# # transpose indices & reshape update values to one dimension
# inputs_size = np.size(inputs)
#
# indices = np.transpose(np.reshape(np.stack([b, y, x, c]), [4, inputs_size]))
# indices = tf.Print(
#     indices,
#     [indices],
#     message="indices:",
#     summarize=2000,
# )
# indices = tf.Print(
#     indices,
#     [b],
#     message="b:",
#     summarize=2000,
# )
# indices = tf.Print(
#     indices,
#     [y],
#     message="y:",
#     summarize=2000,
# )
# values = np.reshape(inputs, [inputs_size])
# # ret = tf.scatter_nd(indices, values, output_shape)
# ret = np.zeros(output_shape)
# print('index_shape:', indices.shape)
# print('values_shape:', values.shape)
# ret[indices] = values

tf.InteractiveSession()
# inputs = np.random.randint(50, size=(1, 6, 6, 3)).astype(np.float32)
# x, idx = tf.nn.max_pool_with_argmax(inputs, (1, 2, 2, 1), (1, 2, 2, 1), padding='SAME')
# print(repr(inputs))
# print(repr(x.eval()))
# print(repr(idx.eval()))

x = np.array([[[[9., 47., 31.],
                [37., 20., 45.],
                [45., 12., 30.],
                [2., 10., 24.],
                [22., 21., 12.],
                [49., 48., 37.]],

               [[33., 17., 20.],
                [8., 19., 22.],
                [24., 2., 21.],
                [18., 37., 26.],
                [24., 28., 46.],
                [11., 12., 15.]],

               [[2., 10., 42.],
                [24., 12., 16.],
                [15., 17., 42.],
                [37., 13., 10.],
                [46., 8., 9.],
                [15., 49., 36.]],

               [[15., 34., 21.],
                [46., 46., 49.],
                [0., 23., 11.],
                [29., 29., 27.],
                [42., 20., 34.],
                [45., 13., 11.]],

               [[40., 12., 22.],
                [27., 14., 37.],
                [2., 5., 45.],
                [19., 28., 27.],
                [4., 48., 44.],
                [25., 1., 20.]],

               [[26., 4., 6.],
                [36., 1., 32.],
                [14., 6., 8.],
                [46., 20., 37.],
                [34., 9., 11.],
                [17., 14., 25.]]]], dtype=np.float32)

inputs = np.array([[[[37., 47., 45.],
                     [45., 37., 30.],
                     [49., 48., 46.]],

                    [[46., 46., 49.],
                     [37., 29., 42.],
                     [46., 49., 36.]],

                    [[40., 14., 37.],
                     [46., 28., 45.],
                     [34., 48., 44.]]]], dtype=np.float32)
mask = np.array([[[[3, 1, 5],

                   [6, 28, 8],
                   [15, 16, 32]],

                  [[57, 58, 59],
                   [45, 64, 44],
                   [48, 52, 53]],

                  [[72, 76, 77],
                   [99, 82, 80],
                   [102, 85, 86]]]])

# input_shape = inputs.shape
# ksize = (1, 2, 2, 1)
# #  calculation new shape
# output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
# # calculation indices for batch, height, width and feature maps
# one_like_mask = np.ones_like(mask)
# batch_range = np.reshape(list(range(output_shape[0])), [input_shape[0], 1, 1, 1])
# # batch_range = tf.Print(
# #     batch_range,
# #     [batch_range],
# #     message="batch_range:",
# #     summarize=2000,
# # )
# b = one_like_mask * batch_range
# y = mask // (output_shape[2] * output_shape[3])
# x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
# channel_range = list(range(output_shape[3]))
# c = one_like_mask * channel_range
# # transpose indices & reshape update values to one dimension
# inputs_size = np.size(inputs)
#
# indices = np.transpose(np.reshape(np.stack([b, y, x, c]), [4, inputs_size]))

# rs = [0] * 108
rs = np.zeros((1, 6, 6, 3)).flatten()
for pair in zip(inputs.reshape(-1), mask.reshape(-1)):
    rs[pair[1]] = pair[0]
rs = rs.reshape(1, 6, 6, 3)
print(rs)


def unpool_with_argmax(
        inputs,
        mask,
        ksize
):
    """Ref: https://github.com/mshunshin/SegNetCMR/blob/master/SegNetCMR/layers.py"""

    input_shape = inputs.get_shape().as_list()

    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    # batch_range = tf.Print(
    #     batch_range,
    #     [batch_range],
    #     message="batch_range:",
    #     summarize=2000,
    # )
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    channel_range = tf.range(output_shape[3], dtype=tf.int64)
    c = one_like_mask * channel_range
    # transpose indices & reshape update values to one dimension
    inputs_size = tf.size(inputs)

    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, c]), [4, inputs_size]))
    # indices = tf.Print(
    #     indices,
    #     [indices],
    #     message="indices:",
    #     summarize=2000,
    # )
    # indices = tf.Print(
    #     indices,
    #     [b],
    #     message="b:",
    #     summarize=2000,
    # )
    # indices = tf.Print(
    #     indices,
    #     [y],
    #     message="y:",
    #     summarize=2000,
    # )
    values = tf.reshape(inputs, [inputs_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret

inputs = tf.convert_to_tensor(inputs)
mask = tf.convert_to_tensor(mask)
rs_tf = unpool_with_argmax(inputs, mask, (1, 2, 2, 1))
print(rs_tf.eval())
print(np.allclose(rs, rs_tf.eval()))
