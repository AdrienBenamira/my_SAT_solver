import torch
import torch.nn as nn

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



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

import torch
from torch.autograd import Function

class L1Penalty(Function):

    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(1)
        grad_input += grad_output
        return grad_input, None


class NeuroSAT(nn.Module):
    def __init__(self, args, device):
        super(NeuroSAT, self).__init__()
        self.args = args
        
        d = self.args.embbeding_dim
        self.d = d
        self.l1weight = args.l1weight

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

        self.device = device



    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        n_probs = len(problem.is_sat)
        # print(n_vars, n_lits, n_clauses, n_probs)

        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()



        init_ts = self.init_ts.to(self.device)
        # 1 x n_lits x dim & 1 x n_clauses x dim
        L_init = self.L_init(init_ts).view(1, 1, -1).to(self.device)
        # print(L_init.shape)
        L_init = L_init.repeat(1, n_lits, 1)
        moitie = int(n_lits/2)
        L_init[:,:moitie,:] = -L_init[:,:moitie,:]
        C_init = self.C_init(init_ts).view(1, 1, -1).to(self.device)
        # print(C_init.shape)
        C_init = C_init.repeat(1, n_clauses, 1)

        # print(L_init.shape, C_init.shape)

        L_state = (L_init, torch.zeros(1, n_lits, self.d).to(self.device))
        C_state = (C_init, torch.zeros(1, n_clauses, self.d).to(self.device))
        L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells),
                                            torch.Size([n_lits, n_clauses])).to_dense().to(self.device)

        # print(ts_L_unpack_indices.shape)

        #
        self.acp_dico = {str(index_T): None for index_T in range(self.args.T)}

        for index_T in range(self.args.T):
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

            if str(index_T) in self.acp_dico.keys():
                self.acp_dico[str(index_T)] = L_state[0].squeeze(0)

            logits = L_state[0].squeeze(0)
            clauses = C_state[0].squeeze(0)

            if self.args.sparse:

                logits = L1Penalty.apply(logits, self.l1weight)
                #clauses = L1Penalty.apply(clauses, self.l1weight)

        if not self.args.sparse:
            logits = L_state[0].squeeze(0)
            #clauses = C_state[0].squeeze(0)

        #logits = L1Penalty.apply(logits, self.l1weight)
        #clauses = L1Penalty.apply(clauses, self.l1weight)

        # print(logits.shape, clauses.shape)

        vote = self.L_vote(logits)

        self.all_votes = vote
        self.final_lits = logits
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


    def find_solutions(self, problem, model, path):
        def flip_vlit(vlit):
            if vlit < problem.n_vars: return vlit + problem.n_vars
            else: return vlit - problem.n_vars

        n_batches = len(problem.is_sat)
        n_vars_per_batch = problem.n_vars // n_batches

        model.eval()

        with torch.set_grad_enabled(False):

            _ = model(problem)

            all_votes = model.all_votes
            final_lits = model.final_lits
            #all_votes, final_lits, _, _ = self.sess.run([self.all_votes, self.final_lits, self.logits, self.predict_costs], feed_dict=d)

            solutions = []
            for batch in range(len(problem.is_sat)):
                decode_cheap_A = (lambda vlit: all_votes[vlit, 0] > all_votes[flip_vlit(vlit), 0])
                decode_cheap_B = (lambda vlit: not decode_cheap_A(vlit))



                def reify(phi):
                    xs = list(zip([phi(vlit) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)],
                                  [phi(flip_vlit(vlit)) for vlit in range(batch * n_vars_per_batch, (batch+1) * n_vars_per_batch)]))
                    def one_of(a, b): return (a and (not b)) or (b and (not a))
                    assert(all([one_of(x[0], x[1]) for x in xs]))
                    return [x[0] for x in xs]

                if self.solves(problem, batch, decode_cheap_A):
                    solutions.append(reify(decode_cheap_A))

                elif self.solves(problem, batch, decode_cheap_B):
                    solutions.append(reify(decode_cheap_B))

                else:

                    L = np.reshape(final_lits.cpu(), [2 * n_batches, n_vars_per_batch, self.d])
                    L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)

                    kmeans = KMeans(n_clusters=2, random_state=0).fit(L)
                    distances = kmeans.transform(L)
                    scores = distances * distances

                    def proj_vlit_flit(vlit):
                        if vlit < problem.n_vars: return vlit - batch * n_vars_per_batch
                        else:                     return ((vlit - problem.n_vars) - batch * n_vars_per_batch) + n_vars_per_batch

                    def decode_kmeans_A(vlit):
                        return scores[proj_vlit_flit(vlit), 0] + scores[proj_vlit_flit(flip_vlit(vlit)), 1] > \
                            scores[proj_vlit_flit(vlit), 1] + scores[proj_vlit_flit(flip_vlit(vlit)), 0]



                    decode_kmeans_B = (lambda vlit: not decode_kmeans_A(vlit))

                    if self.solves(problem, batch, decode_kmeans_A):
                        solutions.append(reify(decode_kmeans_A))

                    elif self.solves(problem, batch, decode_kmeans_B):
                        solutions.append(reify(decode_kmeans_B))

                    else:
                        solutions.append(None)



                    if solutions[-1] is not None:
                        print(solutions[-1])

                        liste_plot = []

                        for key_acp in model.acp_dico.keys():

                            X2 = model.acp_dico[key_acp]

                            liste_plot.append(plot_for_offset(X2,n_batches, n_vars_per_batch, self.d, batch, key_acp, solutions))

                        imageio.mimsave(path + '/ACP_'+str(problem.dimacs[batch])+'.gif', liste_plot, fps=5)

        return solutions

    def solves(self, problem, batch, phi):
        start_cell = sum(problem.n_cells_per_batch[0:batch])
        end_cell = start_cell + problem.n_cells_per_batch[batch]

        if start_cell == end_cell:
            # no clauses
            return 1.0

        current_clause = problem.L_unpack_indices[start_cell, 1]
        current_clause_satisfied = False

        for cell in range(start_cell, end_cell):
            next_clause = problem.L_unpack_indices[cell, 1]

            # the current clause is over, so we can tell if it was unsatisfied
            if next_clause != current_clause:
                if not current_clause_satisfied:
                    return False

                current_clause = next_clause
                current_clause_satisfied = False

            if not current_clause_satisfied:
                vlit = problem.L_unpack_indices[cell, 0]
                #print("[%d] %d" % (batch, vlit))
                if phi(vlit):
                    current_clause_satisfied = True

        # edge case: the very last clause has not been checked yet
        if not current_clause_satisfied: return False
        return True


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio

def plot_for_offset(X2,n_batches, n_vars_per_batch, d, batch, key_acp, solutions):


    L = np.reshape(X2.cpu(), [2 * n_batches, n_vars_per_batch, d])
    L = np.concatenate([L[batch, :, :], L[n_batches + batch, :, :]], axis=0)

    X = L


    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['navy', 'darkorange']
    target_names = ['x', '!x']
    lw = 2

    index_Y = np.array([i for i in range(2 * n_vars_per_batch)])
    y = index_Y < n_vars_per_batch

    for color, i, target_name in zip(colors, [0, 1], target_names):
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set(xlabel='Composante 1', ylabel='Composante 2',
           title='PCA on L at time ' + str(key_acp) + " with answer : " + str(solutions[-1]))

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

kwargs_write = {'fps':1.0, 'quantizer':'nq'}

