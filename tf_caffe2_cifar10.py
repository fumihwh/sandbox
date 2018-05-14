import os

import caffe2.python.onnx.backend as caffe2_bk
import numpy as np
import tensorflow as tf
import onnx
import onnx_tf.backend as tf_bk

with open(os.path.join('pb/onnx_gh_mnist_nchw.pb'), "rb") as f:
    onnx_model = onnx.load(f)

onnx.checker.check_model(onnx_model)
# print(onnx_model)

img = np.random.randn(50, 784).astype(np.float32)

img = (img - np.min(img)) / (np.max(img) - np.min(img))

output_c2 = caffe2_bk.run_model(onnx_model, [img])
# output_tf = tf_bk.run_model(onnx_model, [img])
#
print(output_c2)
# print(output_tf)

# print(np.allclose(output_c2, output_tf))
