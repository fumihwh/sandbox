from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from onnx_tf.backend import run_node
from onnx import helper
import itertools
import nnvm
import onnx
import tvm
from tvm.contrib import graph_runtime

onnx_graph = onnx.load('pb/onnx_reshape_4.pb')
sym, params = nnvm.frontend.from_onnx(onnx_graph)
target = "llvm"
shape_dict = {'input_0': (1, 5, 5, 3)}

graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

data = np.random.randn(1, 5, 5, 3).astype(np.float32)
ctx = tvm.context(target, 0)
m = graph_runtime.create(graph, lib, ctx)
m.set_input('input_0', tvm.nd.array(data))
m.run()

output_shape = (1, 75)
tvm_out = m.get_output(0, tvm.nd.empty(output_shape, 'float32')).asnumpy()
print(tvm_out)
print(np.reshape(data, (1, 75)))
print(np.allclose(np.reshape(data, (1, 75)), tvm_out))
