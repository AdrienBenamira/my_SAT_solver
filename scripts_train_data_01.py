import os
import random

random.seed(0)


#for n_pairs in ["little_dataset_100p/", "dataset_10000p/"]:
n_pairs = "dataset_10000p/"

for limit in ["12_15/"]:


    seed = random.randint(0,2)
    path1 = "./data/"+n_pairs+ limit +"train/pickle"
    path2 = "./data/" + n_pairs + limit + "test/pickle"
    path3 = "./data/" + n_pairs + limit + "val/pickle"


    os.system(
        'python3 main.py --device ' + str(1) + ' --seed ' + str(seed) + ' --initialisation random --sparse No --sparseKL No --train_dir '+path1+
        ' --test_dir '+path2+' --val_dir ' +path3)

    print()
    print()
    print()











