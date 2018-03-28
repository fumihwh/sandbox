import onnx
from caffe2.python.onnx import backend
from onnx_tf.backend import run_model

import numpy as np

def test():
  # input = np.random.randn(1, 5, 5, 3).astype(np.float32)
  input = np.random.randn(1, 784).astype(np.float32)

  # model = onnx.load('pb/onnx_reshape.pb')
  model = onnx.load('pb/onnx_gh_mnist_nchw.pb')
  outputs_caffe2 = backend.run_model(model, [input])
  print(outputs_caffe2)

  outputs_tf = run_model(model, [input])

  # m = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

  print(outputs_tf)

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

# node = onnx.helper.make_node(
#   'AveragePool',
#   inputs=['x'],
#   outputs=['y'],
#   kernel_shape=[3, 3],
#   strides=[2, 2],
#   # auto_pad='SAME_UPPER'
#   # pads=[1, 1, 1, 1]
#   auto_pad='same'
# )
# x = np.array([[[
#   [1, 2, 3, 4, 5],
#   [6, 7, 8, 9, 10],
#   [11, 12, 13, 14, 15],
#   [16, 17, 18, 19, 20],
#   [21, 22, 23, 24, 25],
# ]]]).astype(np.float32)
# y = np.array([[[[4, 5.5, 7],
#                 [11.5, 13, 14.5],
#                 [19, 20.5, 22]]]]).astype(np.float32)
#
# t_x = Variable(torch.from_numpy(x))
# outputs_torch = m(t_x)
# print(outputs_torch)
#
# outputs_caffe2 = backend.run_node(node, [x])
# print(outputs_caffe2['y'])
#
# workspace.ResetWorkspace()
# workspace.FeedBlob("data", x)
# deploy_model = model_helper.ModelHelper("deploy_model")
# brew.average_pool(deploy_model, "data", "result", kernel=3, stride=2, legacy_pad=2)
# workspace.RunNetOnce(deploy_model.param_init_net)
# workspace.CreateNet(deploy_model.net, overwrite=True)
# workspace.RunNetOnce(deploy_model.net)
# result = workspace.FetchBlob("result")
# print(result)

