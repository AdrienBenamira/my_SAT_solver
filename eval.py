from tqdm import tqdm

from src.model_neurosat_v2 import NeuroSAT2
from utils.config import Config
import PyMiniSolvers.minisolvers as minisolvers
import random
import numpy as np
import os
from utils.create_database_random import DataGenerator, ProblemsLoader
from src.model_neurosat import *
from torch.utils.tensorboard import SummaryWriter
from src.trainer import train_model, solve_pb
import json
import argparse
import datetime
from utils.utils import str2bool, dir_path, two_args_str_int
import time


config = Config()

# initiate the parser
parser = argparse.ArgumentParser()


parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[0, 1, 2])
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

parser.add_argument("--train_dir", default=config.path.train_dir)
parser.add_argument("--val_dir", default=config.path.val_dir)
parser.add_argument("--test_dir", default=config.path.test_dir)
parser.add_argument("--logs_tensorboard", default=config.path.logs_tensorboard)
parser.add_argument("--model1", default=config.path.model1, type=dir_path)
parser.add_argument("--model2", default=config.path.model2, type=dir_path)

parser.add_argument("--n_epochs", default=config.training.n_epochs, type=two_args_str_int)
parser.add_argument("--embbeding_dim", default=config.training.embbeding_dim, type=two_args_str_int)
parser.add_argument("--weight_decay", default=config.training.weight_decay, type=two_args_str_int)
parser.add_argument("--lr", default=config.training.lr, type=two_args_str_int)
parser.add_argument("--T", default=config.training.T, type=two_args_str_int)
parser.add_argument("--sparse", default=config.training.sparse, type=str2bool, nargs='?', const=False)
parser.add_argument("--l1weight", default=config.training.l1weight, type=two_args_str_int)
parser.add_argument("--sparseKL", default=config.training.sparseKL, type=str2bool, nargs='?', const=False)
parser.add_argument("--KL_distribval", default=config.training.KL_distribval, type=two_args_str_int)
parser.add_argument("--initialisation", default=config.training.initialisation, choices=['random', 'predict_model'])
parser.add_argument("--initialisation_eval", default=config.eval.initialisation, choices=['random', 'predict_model'])




args = parser.parse_args()


device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
date = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
model_test = args.task_name + "/"
writer = SummaryWriter(args.logs_tensorboard + model_test + date)
path_save_model = args.logs_tensorboard + model_test + date + "/"
with open(path_save_model+'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print("Use Hardware : ", device)

# Reproductibilites
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





train_problems_loader = ProblemsLoader([args.train_dir + "/" + f for f in os.listdir(args.train_dir)])
val_problems_loader = ProblemsLoader([args.val_dir + "/" + f for f in os.listdir(args.val_dir)])
test_problems_loader = ProblemsLoader([args.test_dir + "/" + f for f in os.listdir(args.test_dir)])
dataloaders = {'train': train_problems_loader, 'val': val_problems_loader, 'test': test_problems_loader}


if args.initialisation_eval == "predict_model":
    model = NeuroSAT(args, device)
    net = torch.load(args.model1, map_location=torch.device(device))
    model.load_state_dict(net['state_dict'])
    model.eval()
    problems_test, train_filename = dataloaders["test"].get_next()
    model2 = NeuroSAT2(args, model)
    model2.to(device)
    net = torch.load(args.model2, map_location=torch.device(device))
    model2.load_state_dict(net['state_dict'])
    model2.eval()
    solve_pb(problems_test, model2, path_save_model)
if args.initialisation_eval == "random":
    model = NeuroSAT(args, device)
    net = torch.load(args.model2, map_location=torch.device(device))
    model.load_state_dict(net['state_dict'])
    model.eval()
    problems_test, train_filename = dataloaders["test"].get_next()

    solve_pb(problems_test, model, path_save_model)




