from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from google.protobuf import json_format
from google.protobuf.json_format import MessageToJson
from tensorflow.core.framework import graph_pb2, node_def_pb2
from google.protobuf import text_format
from onnx_tf.frontend import convert_graph
from onnx_tf.backend import run_model
import pathlib
from tensorflow.python.framework import graph_io
from onnx.helper import make_model
import onnx
import ssl
import time
import click

# This restores the same behavior as before.
ssl._create_default_https_context = ssl._create_unverified_context


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te - ts))
        return result

    return timed


def cnn_model_fn(input):
    """Model function for CNN."""
    input_layer = tf.reshape(tensor=input, shape=[-1, 28, 28, 1])

    x = tf.layers.conv2d(
        inputs=input_layer,
        filters=10,
        kernel_size=[3, 3],
        strides=2,
        padding="SAME",
        use_bias=False,
        activation=tf.nn.relu)

    x = tf.layers.conv2d(
        inputs=x,
        filters=20,
        kernel_size=[5, 5],
        strides=2,
        padding="SAME",
        use_bias=False,
        activation=tf.nn.relu)

    # x = tf.contrib.layers.flatten(x)
    x = tf.reshape(x, (-1, 980))
    logits = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu, use_bias=False)

    logits = tf.identity(logits, 'output')
    return logits


def run():
    with tf.Graph().as_default():
        data_placeholder = tf.placeholder(name='input_ph', shape=[None, 784], dtype=tf.float32)
        label_placeholder = tf.placeholder(name='label_ph', shape=[None, ], dtype=tf.int32)
        logits = cnn_model_fn(data_placeholder)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_placeholder, logits=logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        # Load training and eval data
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")

        for _ in range(2):
            batch = mnist.train.next_batch(50)
            sess.run(train_op, feed_dict={data_placeholder: batch[0], label_placeholder: batch[1]})

        minimal_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            ['output'],
        )
        tf.train.write_graph(minimal_graph, 'pb', 'test_2.pb', as_text=False)


@timeit
def front(pb_name):
    graph_def = graph_pb2.GraphDef()

    with open(os.path.join('pb', pb_name), "rb") as f:
        graph_def.ParseFromString(f.read())
        node_def = graph_def.node[-1]

    onnx_gh_def = convert_graph(graph_def, node_def)
    model = make_model(onnx_gh_def)
    f = open(os.path.join('pb', pb_name).replace('tf', 'onnx'), 'wb')
    f.write(model.SerializeToString())
    f.close()


def back(pb_name):
    with open(os.path.join('pb', pb_name), "rb") as f:
        test = onnx.load(f)
        rs = run_model(test, [np.random.randn(5, 1, 3)])
        print(rs)
        pass


def test(suffix):
    onnx_graph = 'onnx' + suffix
    tf_graph = 'tf' + suffix
    input = np.random.randn(10, 784)
    with tf.Session() as sess:
        model_filename = os.path.join('pb', tf_graph)
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with sess.graph.as_default() as graph:
            tf.import_graph_def(
                graph_def,
                name=''
            )
        # saver = tf.train.import_meta_graph(os.path.join('pb', tf_graph))
        # saver.restore(sess)
        output = sess.graph.get_tensor_by_name('output:0')
        # data_placeholder = tf.get_collection('input_ph')[0]
        rs_tf = sess.run(output, feed_dict={sess.graph.get_tensor_by_name('input_ph:0'): input})
        print(rs_tf)

    with open(os.path.join('pb', onnx_graph), "rb") as f:
        test = onnx.load(f)
        rs_onnx = run_model(test, [input])
        print(rs_onnx)

    print(np.allclose(rs_tf, rs_onnx))


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '-c',
    '--command',
    type=click.Choice(['front', 'back', 'test']),
    help='Choose the command from front, back, test.',
)
@click.option(
    '-args',
    multiple=True,
    help='Command\'s args',
)
def main(command, args):
    method = eval(command)
    method(*args)


if __name__ == '__main__':
    main()
