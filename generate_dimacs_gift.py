import argparse

from utils.config import Config
import PyMiniSolvers.minisolvers as minisolvers
import random
import numpy as np
import os
import time
from utils.create_database_random import DataGenerator, ProblemsLoader
import torch
from src.model_neurosat import *
from torch.utils.tensorboard import SummaryWriter
from src.trainer import train_model
from utils.utils import str2bool, dir_path, two_args_str_int
import os
import math
import numpy as np
import random
import argparse
import pickle
from utils.create_database_random import DataGenerator
config = Config()

np.random.seed(0)

#-------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR SAT PB

def une_sbox(numeros_SBOX, L, D):
    """
	numeros_SBOX : 1 a n
	L : Largeur
	D = Depth
	"""
    assert L > 0
    assert D > 0
    n = L * D
    assert numeros_SBOX > 0
    assert numeros_SBOX < n + 1
    index_start = (numeros_SBOX - 1) * 8 + 1
    index_end = 8 + (numeros_SBOX - 1) * 8
    w_index = 8 * L * D + numeros_SBOX
    all_i = [i for i in range(index_start, index_end + 1)]
    all_i_bar = [-1 * i for i in range(index_start, index_end + 1)]
    w_index_bar = -w_index
    x_i = all_i[:4]
    y_i = all_i[4:]
    x_i_b = all_i_bar[:4]
    y_i_b = all_i_bar[4:]
    c1 = [x_i_b[0], x_i_b[1], x_i_b[2], x_i[3], y_i_b[2]]
    c2 = [x_i[0], x_i_b[1], x_i_b[2], x_i_b[3], y_i_b[1], y_i_b[2]]
    c3 = [x_i[0], x_i_b[1], x_i[2], y_i_b[0], y_i[1], y_i_b[2]]
    c4 = [x_i[1], x_i[2], y_i_b[0], y_i_b[1], y_i_b[2], y_i[3]]
    c5 = [x_i[1], x_i_b[2], x_i_b[3], y_i[1], y_i_b[2], y_i[3]]
    c6 = [x_i[0], x_i[1], x_i[2], y_i_b[0], y_i_b[1], y_i[2], y_i_b[3]]
    c7 = [x_i[0], x_i[1], x_i_b[2], x_i_b[3], y_i[1], y_i[2], y_i_b[3]]
    c8 = [y_i_b[0], w_index]
    c9 = [y_i_b[1], w_index]
    c10 = [y_i_b[2], w_index]
    c11 = [y_i_b[3], w_index]
    c12 = [x_i[0], x_i_b[1], x_i_b[2], x_i[3], y_i[2]]
    c13 = [x_i[1], x_i_b[2], x_i[3], y_i_b[2], y_i_b[3]]
    c14 = [x_i_b[1], x_i[3], y_i_b[0], y_i_b[2], y_i_b[3]]
    c15 = [x_i_b[0], x_i_b[1], y_i[0], y_i_b[2], y_i_b[3]]
    c16 = [x_i[0], x_i_b[3], y_i[0], y_i_b[2], y_i_b[3]]
    c17 = [x_i_b[0], x_i[2], x_i[3], y_i[2], y_i_b[3]]
    c18 = [x_i_b[0], y_i[0], y_i_b[1], y_i[2], y_i_b[3]]
    c19 = [x_i[0], x_i_b[1], x_i[2], x_i[3], y_i[3]]
    c20 = [x_i_b[0], x_i[1], x_i[2], x_i[3], y_i[3]]
    c21 = [x_i_b[1], y_i[0], y_i_b[1], y_i_b[2], y_i[3]]
    c22 = [x_i[1], x_i_b[2], x_i[3], y_i[2], y_i[3]]
    c23 = [x_i[1], x_i_b[3], y_i[0], y_i[2], y_i[3]]
    c24 = [x_i[0], y_i[0], y_i_b[1], y_i[2], y_i[3]]
    c25 = [x_i_b[1], y_i[0], y_i[1], y_i[2], y_i[3]]
    c26 = [x_i[0], x_i[1], x_i[2], x_i[3], w_index_bar]
    c27 = [x_i[0], y_i[0], y_i[1], y_i[2], w_index_bar]
    c28 = [x_i[1], y_i[0], y_i[1], y_i[3], w_index_bar]
    c29 = [x_i[0], x_i_b[1], x_i[2], x_i_b[3], y_i[1], y_i_b[3]]
    c30 = [x_i_b[0], x_i[1], x_i_b[3], y_i_b[0], y_i_b[2], y_i_b[3]]
    c31 = [x_i_b[0], x_i[2], y_i_b[0], y_i[1], y_i[2], y_i_b[3]]
    c32 = [x_i_b[0], x_i_b[1], x_i_b[3], y_i_b[0], y_i[2], y_i[3]]
    c33 = [x_i_b[0], x_i[1], x_i_b[2], x_i_b[3], y_i_b[0], y_i_b[1], y_i_b[3]]
    c34 = [x_i_b[1], x_i_b[2], x_i_b[3], y_i_b[0], y_i_b[1], y_i[2], y_i_b[3]]
    c35 = [x_i_b[0], x_i_b[1], x_i[2], x_i_b[3], y_i_b[0], y_i_b[1], y_i[3]]
    c36 = [x_i_b[0], x_i_b[1], x_i_b[2], x_i_b[3], y_i_b[0], y_i[1], y_i[3]]
    print()
    print("SBOX NUMERO:", numeros_SBOX)
    print("ENTREES VARIABLES: ", x_i)
    print("SORTIES VARIABLES: ", y_i)
    print("SBOX W index: ", w_index)
    print()
    return [c1, c2, c3, c4, c5, c6, c7, c8, c9,
            c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
            c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
            c30, c31, c32, c33, c34, c35, c36]


def compteur(L, D, tau):
    """
	numeros_SBOX : 1 a n
	L : Largeur
	D = Depth
    tau = nbre de Sbox inferieure ou egale a tau
	U = matrice of shape (n − 1, τ )
	U[i,j] = u_i,j
	"""

    w_liste = [8 * L * D + i + 1 for i in range(L * D)]
    w_all = np.array(w_liste)
    w_all_bar = -1 * w_all
    n = L * D
    U = np.zeros((tau, n - 1), dtype=np.int)
    U = U.transpose()
    offset = 9 * n
    for i in range(n - 1):
        for j in range(tau):
            U[i, j] = offset + 1
            offset = U[i, j]

    print("VARIABLES W POUR COMPTER NBRE DE SBOX: ", w_all)
    print()
    print("VARIABLES ARTIFICIELLLES U POUR COMPTER NBRE DE SBOX: ", U)
    print()

    U_bar = -1 * U
    c_all = []
    c1 = [w_all_bar[0], U[0, 0]]
    c_all.append(c1)
    for j in range(1, tau):
        c_all.append([U_bar[0, j]])
    for i in range(1, n - 1):
        c_all.append([w_all_bar[i], U[i, 0]])
    for i in range(1, n - 1):
        c_all.append([U_bar[i - 1, 0], U[i, 0]])
    for i in range(1, n - 1):
        for j in range(1, tau):
            c_all.append([w_all_bar[i], U_bar[i - 1, j - 1], U[i, j]])
            c_all.append([U_bar[i - 1, j], U[i, j]])
    for i in range(1, n - 1):
        c_all.append([w_all_bar[i], -1*U[i - 1, tau - 1]])
    c_all.append([w_all_bar[n - 1], -1*U[n - 2, tau - 1]])



    ntot = 1 + (tau-1) + (n-2) + (n-2) + (n-2)*(tau-1) + (n-2)*(tau-1)+ (n-2) + 1

    assert len(c_all) == ntot

    return c_all


def lien_entre_couche_v8bits(D):
    """
	D = Depth
	"""
    for d in range(D-1):
        numeros_SBOX_all = [1 + 2 * d, 2 + 2 * d, 3 + 2 * d, 4 + 2 * d]
        index_all = []
        index_all2 = []
        for l in range(2):
            numeros_SBOX = numeros_SBOX_all[l]
            index_start = (numeros_SBOX - 1) * 8 + 1 + 4
            index_end = index_start + 4
            index_all += [index_actu for index_actu in range(index_start, index_end)]
        for l in range(2):
            numeros_SBOX = numeros_SBOX_all[l + 2]
            index_start = (numeros_SBOX - 1) * 8 + 1
            index_end = index_start + 4
            index_all2 += [index_actu for index_actu in range(index_start, index_end)]

    print("VARIABBLES PERMUTATIONS ENTREES: ", index_all)
    print("VARIABBLES PERMUTATIONS SORTIES:", index_all2)
    print()

    c = []
    for j in range(len(index_all)):
        index_autre = (j + 2) % len(index_all)
        c += [[index_all[j], index_all2[index_autre]]]
        c += [[-index_all[j], -index_all2[index_autre]]]
    return c


def lien_entre_couche_v16bits(D):
    """
	D = Depth
	"""
    for d in range(D-1):
        numeros_SBOX_all = [1 + 2 * d, 2 + 2 * d, 3 + 2 * d, 4 + 2 * d,
                            5 + 2 * d, 6 + 2 * d, 7 + 2 * d, 8 + 2 * d]
        index_all = []
        index_all2 = []
        for l in range(4):
            numeros_SBOX = numeros_SBOX_all[l]
            index_start = (numeros_SBOX - 1) * 8 + 1 + 4
            index_end = index_start + 4
            index_all += [index_actu for index_actu in range(index_start, index_end)]
        for l in range(4):
            numeros_SBOX = numeros_SBOX_all[l + 4]
            index_start = (numeros_SBOX - 1) * 8 + 1
            index_end = index_start + 4
            index_all2 += [index_actu for index_actu in range(index_start, index_end)]

    print("VARIABBLES PERMUTATIONS ENTREES: ", index_all)
    print("VARIABBLES PERMUTATIONS SORTIES:", index_all2)
    print()

    c = []
    compteur = -1
    for j in range(int(len(index_all)/4)):
        for k in range(4):
            compteur +=1
            index_autre = j + k*4
            c += [[index_all[compteur], index_all2[index_autre]]]
            c += [[-index_all[compteur], -index_all2[index_autre]]]

    return c


def input_output_egalite(x, y, L, D):
    """
	numeros_SBOX : 1 a n
	L : Largeur
	D = Depth
	"""
    numeros_SBOX_all = []
    for l in range(L):
        numeros_SBOX_all.append(l+1)
    for l in range(L):
        numeros_SBOX_all.append(l+1 + (D-1)*L)
    index_all_entry = []
    index_all_exit = []
    for numeros_SBOX in numeros_SBOX_all[:L]:
        index_start = (numeros_SBOX - 1) * 8 + 1
        index_end = 4 + index_start
        index_all_entry += [index_actu for index_actu in range(index_start, index_end)]
    for numeros_SBOX in numeros_SBOX_all[L:]:
        index_start = (numeros_SBOX - 1) * 8 + 5
        index_end = index_start + 4
        index_all_exit += [index_actu for index_actu in range(index_start, index_end)]
    c=[]
    for index_x, xi in enumerate(x):
        if xi:
            c+=[[index_all_entry[index_x]]]
        else:
            c += [[-index_all_entry[index_x]]]
    for index_y, yi in enumerate(y):
        if yi:
            c+=[[index_all_exit[index_y]]]
        else:
            c += [[-index_all_exit[index_y]]]





    return(c)


#-------------------------------------------------------------------------------------------------------------------
# FUNCTION FOR CIPHER GIFT 16 BITS

S_box = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]
inverseS_box = [0xd, 0x0, 0x8, 0x6, 0x2, 0xc, 0x4, 0xb, 0xe, 0x7, 0x1, 0xa, 0x3, 0x9, 0xf, 0x5]



def substitution(p):
    for i in range(len(p)):
        p[i] = S_box[p[i]]
    return p

def inverseSubstitution(p):
    for i in range(len(p)):
        p[i] = inverseS_box[p[i]]
    return p

def diffusion(p):
    b = [0, 0, 0, 0]
    b[0] = ((p[0] >> 0) & 0x8) ^ ((p[1] >> 1) & 0x4) ^ ((p[2] >> 2) & 0x2) ^ ((p[3] >> 3) & 0x1);
    b[1] = ((p[0] << 1) & 0x8) ^ ((p[1] >> 0) & 0x4) ^ ((p[2] >> 1) & 0x2) ^ ((p[3] >> 2) & 0x1);
    b[2] = ((p[0] << 2) & 0x8) ^ ((p[1] << 1) & 0x4) ^ ((p[2] >> 0) & 0x2) ^ ((p[3] >> 1) & 0x1);
    b[3] = ((p[0] << 3) & 0x8) ^ ((p[1] << 2) & 0x4) ^ ((p[2] << 1) & 0x2) ^ ((p[3] >> 0) & 0x1);
    for i in range(len(p)):
        p[i] = b[i]
    return p


def inverseDiffusion(p):
    b = [0, 0, 0, 0]
    b[0] = ((p[0] >> 0) & 0x8) ^ ((p[1] >> 1) & 0x4) ^ ((p[2] >> 2) & 0x2) ^ ((p[3] >> 3) & 0x1)
    b[1] = ((p[0] << 1) & 0x8) ^ ((p[1] >> 0) & 0x4) ^ ((p[2] >> 1) & 0x2) ^ ((p[3] >> 2) & 0x1)
    b[2] = ((p[0] << 2) & 0x8) ^ ((p[1] << 1) & 0x4) ^ ((p[2] >> 0) & 0x2) ^ ((p[3] >> 1) & 0x1)
    b[3] = ((p[0] << 3) & 0x8) ^ ((p[1] << 2) & 0x4) ^ ((p[2] << 1) & 0x2) ^ ((p[3] >> 0) & 0x1)
    for i in range(len(p)):
        p[i] = b[i]
    return p


def add_contants(x, c):
    x[0] = x[0] ^ c
    return x


def addRoundKey(c, key):
    for i in range(len(c)):
        c[i] = c[i] ^ key[i]
    return c


def Encrypt(c, key, nr):
    c = addRoundKey(c, key)
    for i in range(nr):
        c = substitution(c)
        c = diffusion(c)
        c = add_contants(c, i)
        c = addRoundKey(c, key)
    return c


def Decrypt(c, key, nr):
    c = addRoundKey(c, key)
    for i in range(nr):
        c = add_contants(c, i)
        c = inverseDiffusion(c)
        c = inverseSubstitution(c)
        c = addRoundKey(c, key)
    return c

def urandom_from_random(rng, length):
    if length == 0:
        return b''

    import sys
    chunk_size = 65535
    chunks = []
    while length >= chunk_size:
        chunks.append(rng.getrandbits(
                chunk_size * 4).to_bytes(chunk_size, sys.byteorder))
        length -= chunk_size
    if length:
        chunks.append(rng.getrandbits(
                length * 4).to_bytes(length, sys.byteorder))
    result = b''.join(chunks)
    return result
#-------------------------------------------------------------------------------------------------------------------
#MAIN

L = 2                               #largeur
D = 2                              #Profondeur
nr = D-1                            #Nbre de round
n = L * D                           #nbre totales de sbox
tau = 1                            #Nbre max de Sbox active

assert tau < n + 1

add_confition_input_output = False

nbre_sample_bin = np.random.randint(0, 2 ** L, np.random.randint(1, 16, 1))

print(nbre_sample_bin)

if not add_confition_input_output:
    initialisation = []
    for nbre_sample_bin_u_index, nbre_sample_bin_u in enumerate(nbre_sample_bin):
        initialisation.append([nbre_sample_bin_u + 1])
print(initialisation)

for nbre_sample_tau in range(1, n):

    res_all = []


    for sbox in range(L * D):
        res_all += une_sbox(sbox + 1, L, D)
    res_all += compteur(L, D, nbre_sample_tau)
    if D >1:
        if L == 4:
            res_all += lien_entre_couche_v16bits(D)
        elif L ==2 and not add_confition_input_output:
            res_all += lien_entre_couche_v8bits(D)
        else:
            print("error")

    if add_confition_input_output:
        ct0 = np.random.randint(0, 15, L)
        key = np.random.randint(0, 15, L)
        ct0_origin = ct0.copy()
        ctc0 = Encrypt(ct0, key, nr);
        ct0_bin = []
        for i in range(4):
            ct0_bin += [int(x) for x in list('{:04b}'.format(ct0_origin[i]))]
        ctc0_bin = []
        for i in range(4):
            ctc0_bin += [int(x) for x in list('{:04b}'.format(ctc0[i]))]
        res_all += input_output_egalite(ct0_bin, ctc0_bin, L, D)
    else:
        res_all += initialisation


    max_all = [max(cluase) for cluase in res_all]

    print("NBRE DE CLAUSES: ", len(res_all))
    print("NBRE DE VARIABLES: ", max(max_all))
    n_vars = max(max_all)
    print("RATIO : ", len(res_all) / max(max_all))


    import PyMiniSolvers.minisolvers as minisolvers
    solver = minisolvers.MinisatSolver()
    for i in range(max(max_all)): solver.new_var(dvar=True)
    for iclause in res_all:
        solver.add_clause(iclause)

    is_sat = solver.solve()
    print("IS THE PROBLEM SAT ? ", is_sat)
    if is_sat:
        print()
        print(list(solver.get_model()))


    def write_dimacs_to(n_vars, res_all, out_filename):
        with open(out_filename, 'w') as f:
            f.write("p cnf %d %d\n" % (n_vars, len(res_all)))
            for c in res_all:
                for x in c:
                    f.write("%d " % x)
                f.write("0\n")

    out_filename = "data/test_dimacs/test_"+str(nbre_sample_tau)+".txt"
    write_dimacs_to(n_vars, res_all, out_filename)

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int)
parser.add_argument("--task_name", default=config.general.task_name)
parser.add_argument("--device", default=config.general.device, type=two_args_str_int, choices=[0, 1, 2, 3])
parser.add_argument("--nbre_plot", default=config.general.nbre_plot, type=two_args_str_int)

parser.add_argument("--do_it", default=config.generate_data.do_it, type=str2bool, nargs='?', const=False)
parser.add_argument("--n_pairs", default=config.generate_data.n_pairs, type=two_args_str_int)
parser.add_argument("--min_n", default=config.generate_data.min_n, type=two_args_str_int)
parser.add_argument("--max_n", default=config.generate_data.max_n, type=two_args_str_int)
parser.add_argument("--max_nodes_per_batch", default=config.generate_data.max_nodes_per_batch, type=two_args_str_int)
parser.add_argument("--one", default=config.generate_data.one, type=two_args_str_int)
parser.add_argument("--p_k_2", default=config.generate_data.p_k_2, type=two_args_str_int)
parser.add_argument("--p_geo", default=config.generate_data.p_geo, type=two_args_str_int)

parser.add_argument("--dimacs_dir", default=config.path.dimacs_dir)
parser.add_argument("--out_dir", default=config.path.out_dir)
parser.add_argument("--train_dir", default=config.path.train_dir)
parser.add_argument("--val_dir", default=config.path.val_dir)
parser.add_argument("--test_dir", default=config.path.test_dir)
parser.add_argument("--logs_tensorboard", default=config.path.logs_tensorboard)

args = parser.parse_args()



args.dimacs_dir = "data/test_dimacs"
args.out_dir = "data/test_pickle"

dg = DataGenerator(args, minisolvers)

problems = []
batches = []
n_nodes_in_batch = 0

filenames = os.listdir(dg.config.dimacs_dir)

# to improve batching
filenames = sorted(filenames)

prev_n_vars = None

for filename in filenames:
    #print(filename)
    n_vars, iclauses = dg.parse_dimacs("%s/%s" % (dg.config.dimacs_dir, filename))
    n_clauses = len(iclauses)
    n_cells = sum([len(iclause) for iclause in iclauses])

    n_nodes = 2 * n_vars + n_clauses
    if n_nodes > dg.config.max_nodes_per_batch:
        continue

    batch_ready = False
    if (dg.config.one and len(problems) > 0):
        batch_ready = True
    elif (prev_n_vars and n_vars != prev_n_vars):
        batch_ready = True
    elif (not dg.config.one) and n_nodes_in_batch + n_nodes > dg.config.max_nodes_per_batch:
        batch_ready = True

    if batch_ready:
        batches.append(dg.mk_batch_problem(problems))
        print("batch %d done (%d vars, %d problems)...\n" % (len(batches), prev_n_vars, len(problems)))
        del problems[:]
        n_nodes_in_batch = 0

    prev_n_vars = n_vars

    is_sat, stats = dg.solve_sat(n_vars, iclauses)
    #print(filename, n_vars, iclauses, is_sat)

    problems.append((filename, n_vars, iclauses, is_sat))
    n_nodes_in_batch += n_nodes

if len(problems) > 0:
    batches.append(dg.mk_batch_problem(problems))
    print("batch %d done (%d vars, %d problems)...\n" % (len(batches), n_vars, len(problems)))
    del problems[:]

# create directory
if not os.path.exists(dg.config.out_dir):
    os.mkdir(dg.config.out_dir)

dataset_filename = dg.mk_dataset_filename(dg.config, len(batches))
print("Writing %d batches to %s...\n" % (len(batches), dataset_filename))
with open(dataset_filename, 'wb') as f_dump:
    pickle.dump(batches, f_dump)
