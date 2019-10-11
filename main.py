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
if config.generate_data.do_it:
    dg = DataGenerator(config, minisolvers)
    dg.run_main()


train_problems_loader = ProblemsLoader([config.path.train_dir + "/" + f for f in os.listdir(config.path.train_dir)])
#val_problems_loader = ProblemsLoader([config.path.val_dir + "/" + f for f in os.listdir(config.path.val_dir)])
#test_problems_loader = ProblemsLoader([config.path.test_dir + "/" + f for f in os.listdir(config.path.test_dir)])
dataloaders = {x: train_problems_loader
              for x in ['train', 'val']}

model = NeuroSat(config)
criterion = nn.BCEWithLogitsLoss() #idem à sigmoid_cross_entropy_with_logits from TensorFlow.
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

best_model = train_model(config, writer, model, dataloaders, criterion, optimizer,device, num_epochs=config.training.n_epochs)
