import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuroSat(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.n_vars = XXX
        self.LC_msg = nn.Linear(XXX, XXX)
        self.C_update = nn.LSTM(XXX, XXX)
        self.CL_msg = nn.Linear(XXX, XXX)
        self.L_update = nn.LSTM(XXX, XXX)

    def main_loop(self, L_state, C_state, L_unpack):
        # XXX
        LC_pre_msgs = self.LC_msg(L_state)
        LC_msgs = torch.matmul(L_unpack, LC_pre_msgs, adjoint_a=True)
        _, C_state = self.C_update(inputs=LC_msgs, state=C_state)
        # XXX
        CL_pre_msgs = self.CL_msg(C_state)
        CL_msgs = torch.matmul(L_unpack, CL_pre_msgs)
        new_CL_msgs = tf.concat([CL_msgs, self.flip(L_state)]
        _, L_state = self.L_update(inputs=new_CL_msgs, state=L_state)
        return L_state, C_state

    def flip(self, x):
        #XXX
        return tf.concat([x[self.n_vars:(2*self.n_vars), :], x[0:self.n_vars, :]], axis=0)
        return x

net = Net()
