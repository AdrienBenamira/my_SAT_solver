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


config = Config()


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

writer = SummaryWriter(args.logs_tensorboard)
device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
print()
print("Use Hardware : ", device)
#Reproductibilites
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#Generation de données
dg = DataGenerator(args, minisolvers)
dg.run_main()
