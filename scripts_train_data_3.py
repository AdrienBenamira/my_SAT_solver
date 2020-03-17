import os
import random

random.seed(0)


#for n_pairs in ["little_dataset_100p/", "dataset_10000p/"]:
n_pairs = "dataset_10000p/"

for limit in ["3_5/", "7_10/", "12_15/"]:


    seed = random.randint(0,2)
    path1 = "./data/"+n_pairs+ limit +"train/pickle"
    path2 = "./data/" + n_pairs + limit + "test/pickle"
    path3 = "./data/" + n_pairs + limit + "val/pickle"




    os.system('python3 main.py --device ' + str(3) + ' --seed ' + str(seed) + ' --initialisation random --sparse Yes --sparseKL No --l1weight 0.0001 --train_dir '+path1+
        ' --test_dir '+path2+' --val_dir ' +path3)

    print()
    print()
    print()

    os.system(
        'python3 main.py --device ' + str(3) + ' --seed ' + str(seed) + ' --initialisation random --sparse No --sparseKL Yes --KL_distribval 0.01 --train_dir '+path1+
        ' --test_dir '+path2+' --val_dir ' +path3)








