from utils.config import Config
import PyMiniSolvers.minisolvers as minisolvers
import random
import numpy as np
import os
import time
from utils.generate_data import DataGenerator, ProblemsLoader
import torch
from src.model_neurosat import *
from torch.utils.tensorboard import SummaryWriter
from src.trainer import train_model


config = Config()
writer = SummaryWriter(config.path.logs_tensorboard)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print()
print("Use Hardware : ", device)
#Reproductibilites
random.seed(config.general.seed)
np.random.seed(config.general.seed)
torch.manual_seed(config.general.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#Generation de données
dg = DataGenerator(config, minisolvers)
dg.run_main()
