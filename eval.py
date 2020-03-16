from tqdm import tqdm

from utils.config import Config
import PyMiniSolvers.minisolvers as minisolvers
import random
import numpy as np
import os
from utils.generate_data import DataGenerator, ProblemsLoader
from src.model_neurosat import *
from torch.utils.tensorboard import SummaryWriter
from src.trainer import train_model
import json
import argparse
import datetime
from utils.utils import str2bool, dir_path, two_args_str_int

config = Config()

# initiate the parser
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[0, 1, 2])
parser.add_argument("--task_name", default=config.general.task_name)

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
parser.add_argument("--model", default=config.path.model)

parser.add_argument("--n_epochs", default=config.training.n_epochs, type=two_args_str_int)
parser.add_argument("--embbeding_dim", default=config.training.embbeding_dim, type=two_args_str_int)
parser.add_argument("--weight_decay", default=config.training.weight_decay, type=two_args_str_int)
parser.add_argument("--lr", default=config.training.lr, type=two_args_str_int)
parser.add_argument("--T", default=config.training.T, type=two_args_str_int)
parser.add_argument("--sparse", default=config.training.sparse, type=str2bool, nargs='?', const=False)


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
dataloaders = {'train': train_problems_loader, 'val': val_problems_loader, 'test': test_problems_loader,}

model = NeuroSAT(args)
net = torch.load(args.model)
model.load_state_dict(net['state_dict'])

model.eval()

for situation in dataloaders.keys():
    problems_test, train_filename = dataloaders["test"].get_next()
    test_bar = tqdm(problems_test)
    compteur = 0.0
    total = 0.0
    for _, problem in enumerate(test_bar):
        solutions = model.find_solutions(problem, model, path_save_model)
        for batch, solution in enumerate(solutions):
            total += 1
            if solution is not None:
                print("[%s] %s" % (problem.dimacs[batch], str(solution)))
                compteur +=1

    print('Pourcentage pb solved variables: {:4f}'.format(compteur/total))