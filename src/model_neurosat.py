import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class NeuroSat(nn.Module):
    def __init__(self, config):
        super(NeuroSat, self).__init__()
        self.config = config
        d = config.training.embbeding_dim
        #TODO : changer3 couches

        self.LC_msg = nn.Linear(d, d)
        self.C_update = nn.LSTM(d, d)
        self.CL_msg = nn.Linear(d, d)
        self.L_update = nn.LSTM(2*d, d)
        self.L_vote = nn.Linear(d, 1)


    def main_loop(self, L_state, C_state, hidden_L, hidden_C, M, n_vars):
        self.n_vars = n_vars
        for t in range(self.config.training.T):
            # XXX
            LC_pre_msgs = self.LC_msg(L_state)
            M_batch = M#.unsqueeze(0)
            LC_msgs = torch.matmul(M.t(), LC_pre_msgs)
            C_state, (hidden_C, hidden_C) = self.C_update(LC_msgs, (hidden_C, hidden_C))
            # XXX
            CL_pre_msgs = self.CL_msg(C_state)
            CL_msgs = torch.matmul(M, CL_pre_msgs)
            L_state_flip = self.flip(L_state)
            new_CL_msgs = torch.cat([CL_msgs,L_state_flip], 2)
            L_state, (hidden_L, hidden_L) = self.L_update(new_CL_msgs, (hidden_L, hidden_L))
        return L_state

    def vote(self, L_state, n_batches,n_vars_per_batch):
        all_votes = self.L_vote(L_state)
        all_votes =all_votes.squeeze(0)
        all_votes_join = torch.cat([all_votes[:self.n_vars], all_votes[self.n_vars:]], 1) # n_vars x 2
        all_votes_batched = torch.reshape(all_votes_join, [n_batches, n_vars_per_batch, 2])
        return all_votes_batched

    def flip(self, x):
        #XXX
        return torch.cat([x[self.n_vars:(2*self.n_vars), :], x[0:self.n_vars, :]], 0)

    def final_reducer(self, x):
        reducer = self.config.training.final_reducer
        if reducer == "min":
            return torch.min(torch.mean(x, 2), 1)
        elif reducer == "mean":
            return torch.mean(torch.mean(x, 2), 1)
        elif reducer == "sum":
            return torch.sum(torch.mean(x, 2), 1)
        elif reducer == "max":
            return torch.max(torch.mean(x, 2), 1)
        else:
            raise Exception("Expecting min, mean, or max")
