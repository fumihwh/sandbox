import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision
import onnx
import numpy as np
import os
from onnx_tf.backend import run_model


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#     def forward(self, x):
#         return x.repeat(3, 4, 5, 6)
#
# net = Net()
#
# dummy_input = Variable(torch.randn(2, 3))
#
# output = net.forward(dummy_input)
#
# torch.onnx.export(Net(), dummy_input, "pb/torch_tile.pb", verbose=False, export_params=True)
#
# print(output)



class LSTM_Helper():
    def __init__(self, **params):
        #LSTM Input Names
        X = 'X'
        W = 'W'
        R = 'R'
        B = 'B'
        H_0 = 'initial_h'
        C_0 = 'initial_c'
        P = 'P'
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        num_directions = params[W].shape[0]

        if(num_directions == 1):
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[0] if len(params[X].shape) == 2 else params[X].shape[1]

            b = params[B] if B in params else np.zeros(2 * number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
        else:
            raise NotImplementedError()

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def g(self, x):
        return np.tanh(x)

    def h(self, x):
        return np.tanh(x)

    # def step(self):
    #     h_list = []
    #     for x in np.split(self.X, self.X.shape[0], axis=0):
    #         i = self.f(np.dot(x, np.transpose(w_i)) + np.dot(self.H_0, r_i) + w_bi + r_bi + p_i * self.C_0)
    #         f = self.f(np.dot(x, np.transpose(w_f)) + np.dot(self.H_0, r_f) + w_bf + r_bf + p_f * self.C_0)
    #         c = self.g(np.dot(x, np.transpose(w_c)) + np.dot(self.H_0, r_c) + w_bc + r_bc)
    #         C = f * self.C_0 + i * c
    #         o = self.f(np.dot(x, np.transpose(w_o)) + np.dot(self.H_0, r_o) + w_bo + r_bo + p_o * C)
    #         H = o * self.h(C)
    #         h_list.append(H)
    #         self.H_0 = H
    #         self.C_0 = C
    #     return np.concatenate(h_list)

    def step(self):
        h_list = []
        for x in np.split(self.X, self.X.shape[0], axis=0):
            H, C = self._step_helper(f_func=self.f,
                                     g_func=self.g,
                                     h_func=self.h,
                                     W=self.W,
                                     R=self.R,
                                     P=self.P,
                                     B=self.B,
                                     x=x,
                                     h_0=self.H_0,
                                     c_0=self.C_0,
                                     )
            self.H_0 = H
            self.C_0 = C
            h_list.append(H)
        return np.concatenate(h_list)

    @staticmethod
    def _step_helper(f_func, g_func, h_func, W, R, P, B, x, h_0, c_0):
        # [w_i, w_o, w_f, w_c] = np.split(W, 4)
        # [r_i, r_o, r_f, r_c] = np.split(R, 4)
        [p_i, p_o, p_f] = np.split(P, 3)
        # [w_bi, w_bo, w_bf, w_bc, r_bi, r_bo, r_bf, r_bc] = np.split(B, 8)
        gates = np.dot(x, np.transpose(W)) + np.dot(h_0, np.transpose(R)) + np.add(*np.split(B, 2))
        i, o, f, c = np.split(gates, 4, -1)
        i = f_func(i + p_i * c_0)
        f = f_func(f + p_f * c_0)
        c = g_func(c)
        C = f * c_0 + i * c
        o = f_func(o + p_o * C)
        H = o * h_func(C)
        return H, C


rnn = nn.LSTM(3, 5, 1, )
input = Variable(torch.randn(2, 4, 3))
with torch.no_grad():
    output = rnn(input)
torch.onnx.export(rnn, input, 'pb/torch_lstm.pb')
# print(output)


# x = np.array(input)
# w = np.array(rnn.weight_ih_l0.data)
# r = np.array(rnn.weight_hh_l0.data)
# w_b = np.array(rnn.bias_ih_l0.data)
# r_b = np.array(rnn.bias_hh_l0.data)

x = np.array(input)
[w_i, w_f, w_g, w_o] = np.split(np.array(rnn.weight_ih_l0.data), 4)
[r_i, r_f, r_g, r_o] = np.split(np.array(rnn.weight_hh_l0.data), 4)
w = np.concatenate([w_i, w_o, w_f, w_g])
r = np.concatenate([r_i, r_o, r_f, r_g])

[w_b_i, w_b_f, w_b_g, w_b_o] = np.split(np.array(rnn.bias_ih_l0.data), 4)
[r_b_i, r_b_f, r_b_g, r_b_o] = np.split(np.array(rnn.bias_hh_l0.data), 4)
w_b = np.concatenate([w_b_i, w_b_o, w_b_f, w_b_g])
r_b = np.concatenate([r_b_i, r_b_o, r_b_f, r_b_g])


print(repr(x))
print('w', repr(w))
print('r', repr(r))
print('w_b', repr(w_b))
print('r_b', repr(r_b))
print(repr(np.array(output[0])))

# w = np.array(rnn.weight_ih_l0.data)
# r = np.array(rnn.weight_hh_l0.data)
# w_b = np.array(rnn.bias_ih_l0.data)
# r_b = np.array(rnn.bias_hh_l0.data)

lstm = LSTM_Helper(X=x,
                   W=np.expand_dims(w, 0),
                   R=np.expand_dims(r, 0),
                   B=np.expand_dims(np.concatenate((w_b, r_b)), 0),)
                   # initial_c=init_c1,
                   # initial_h=init_h1)
output_np = lstm.step()
# print(output_np)

print(np.allclose(np.array(output[0]), output_np))
