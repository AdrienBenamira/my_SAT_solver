import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.l3 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    x = self.l1(x)
    x = self.l2(x)
    x = self.l3(x)

    return x


class NeuroSAT(nn.Module):
    def __init__(self, args):
        super(NeuroSAT, self).__init__()
        self.args = args
        
        d = self.args.training.embbeding_dim
        self.d = d

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, d)
        self.C_init = nn.Linear(1, d)

        self.L_msg = MLP(self.d, self.d, self.d)
        self.C_msg = MLP(self.d, self.d, self.d)

        self.L_update = nn.LSTM(self.d * 2, self.d)
        # self.L_norm   = nn.LayerNorm(self.d)
        self.C_update = nn.LSTM(self.d, self.d)
        # self.C_norm   = nn.LayerNorm(self.d)

        self.L_vote = MLP(self.d, self.d, 1)

        self.denom = torch.sqrt(torch.Tensor([self.d]))

    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        n_probs = len(problem.is_sat)
        # print(n_vars, n_lits, n_clauses, n_probs)

        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()

        init_ts = self.init_ts
        # 1 x n_lits x dim & 1 x n_clauses x dim
        L_init = self.L_init(init_ts).view(1, 1, -1)
        # print(L_init.shape)
        L_init = L_init.repeat(1, n_lits, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        # print(C_init.shape)
        C_init = C_init.repeat(1, n_clauses, 1)

        # print(L_init.shape, C_init.shape)

        L_state = (L_init, torch.zeros(1, n_lits, self.d))
        C_state = (C_init, torch.zeros(1, n_clauses, self.d))
        L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells),
                                            torch.Size([n_lits, n_clauses])).to_dense()

        # print(ts_L_unpack_indices.shape)

        for _ in range(self.args.training.T):
            # n_lits x dim
            L_hidden = L_state[0].squeeze(0)
            L_pre_msg = self.L_msg(L_hidden)
            # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
            LC_msg = torch.matmul(L_unpack.t(), L_pre_msg)
            # print(L_hidden.shape, L_pre_msg.shape, LC_msg.shape)

            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)
            # print('C_state',C_state[0].shape, C_state[1].shape)

            # n_clauses x dim
            C_hidden = C_state[0].squeeze(0)
            C_pre_msg = self.C_msg(C_hidden)
            # (n_lits x n_clauses) x (n_clauses x dim) = n_lits x dim
            CL_msg = torch.matmul(L_unpack, C_pre_msg)
            # print(C_hidden.shape, C_pre_msg.shape, CL_msg.shape)

            _, L_state = self.L_update(
                torch.cat([CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state)
            # print('L_state',C_state[0].shape, C_state[1].shape)

        logits = L_state[0].squeeze(0)
        clauses = C_state[0].squeeze(0)

        # print(logits.shape, clauses.shape)
        vote = self.L_vote(logits)
        # print('vote', vote.shape)
        vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1)
        # print('vote_join', vote_join.shape)
        self.vote = vote_join
        vote_join = vote_join.view(n_probs, -1, 2).view(n_probs, -1)
        vote_mean = torch.mean(vote_join, dim=1)
        # print('mean', vote_mean.shape)
        return vote_mean

    def flip(self, msg, n_vars):
        return torch.cat([msg[n_vars:2 * n_vars, :], msg[:n_vars, :]], dim=0)

