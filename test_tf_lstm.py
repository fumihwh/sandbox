import numpy as np
import tensorflow as tf

inputs = np.random.rand(2, 3, 10).astype(np.float32)
sess = tf.InteractiveSession()

x = tf.convert_to_tensor(inputs)
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(20)
cell_fw = tf.nn.rnn_cell.MultiRNNCell([rnn_cell])
bw_rnn_cell = tf.nn.rnn_cell.BasicRNNCell(20)
cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_rnn_cell])
output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, time_major=True, dtype=tf.float32)
# output, state = tf.nn.dynamic_rnn(cell_fw, x, time_major=True, dtype=tf.float32)

sess.run(tf.global_variables_initializer())
print(output[0].eval())
print(output[1].eval())

output_fw = np.expand_dims(output[0].eval(), 1)
output_bw = np.expand_dims(output[1].eval(), 1)

rs = np.concatenate((output_fw, output_bw), axis=1)
print(rs)
print(rs.shape)
print(rs.shape)