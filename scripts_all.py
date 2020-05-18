import os
import random

random.seed(0)
path1 = "./data/dataset_10000p/10_40/train/dimacs"
path2 = "./data/dataset_10000p/10_40/train/pickle"
os.system('python3 generate_data.py --dimacs_dir '+path1+' --out_dir '+path2)

path1 = "./data/dataset_10000p/10_40/val/dimacs"
path2 = "./data/dataset_10000p/10_40/val/pickle"
os.system('python3 generate_data.py --dimacs_dir '+path1+' --out_dir '+path2)

path1 = "./data/dataset_10000p/10_40/test/dimacs"
path2 = "./data/dataset_10000p/10_40/test/pickle"
os.system('python3 generate_data.py --dimacs_dir '+path1+' --out_dir '+path2)

os.system('python3 split.py')


path1 = "./data/mini_GIFT/train/dimacs"
path2 = "./data/mini_GIFT/train/pickle"
os.system('python3 generate_data.py --dimacs_dir '+path1+' --out_dir '+path2)

path1 = "./data/mini_GIFT/val/dimacs"
path2 = "./data/mini_GIFT/val/pickle"
os.system('python3 generate_data.py --dimacs_dir '+path1+' --out_dir '+path2)

path1 = "./data/mini_GIFT/test/dimacs"
path2 = "./data/mini_GIFT/test/pickle"
os.system('python3 generate_data.py --dimacs_dir '+path1+' --out_dir '+path2)

path1 = "./data/dataset_10000p/10_40/train/pickle"
path2 = "./data/dataset_10000p/10_40/val/pickle"
path3 = "./data/dataset_10000p/10_40/test/pickle"

path4= "./data/mini_GIFT/train/pickle"
path5= "./data/mini_GIFT/val/pickle"
path6= "./data/mini_GIFT/test/pickle"
os.system('python3 main.py --cuda 0 --train_dir '+path1+' --val_dir '+path2+' --test_dir '+ path3 +' & python3 main.py --cuda 1 --train_dir '+path4+' --val_dir '+path5+' --test_dir '+path6)