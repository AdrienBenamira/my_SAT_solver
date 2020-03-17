import os
import random

random.seed(0)

os.system('rm -r ./data')
os.system('mkdir ./data')
os.system('mkdir ./data/little_dataset_100p')
os.system('mkdir ./data/little_dataset_100p/3_5')
os.system('mkdir ./data/little_dataset_100p/3_5/train')
os.system('mkdir ./data/little_dataset_100p/3_5/val')
os.system('mkdir ./data/little_dataset_100p/3_5/test')
os.system('mkdir ./data/little_dataset_100p/7_10')
os.system('mkdir ./data/little_dataset_100p/7_10/train')
os.system('mkdir ./data/little_dataset_100p/7_10/val')
os.system('mkdir ./data/little_dataset_100p/7_10/test')
os.system('mkdir ./data/little_dataset_100p/12_15')
os.system('mkdir ./data/little_dataset_100p/12_15/train')
os.system('mkdir ./data/little_dataset_100p/12_15/val')
os.system('mkdir ./data/little_dataset_100p/12_15/test')
os.system('mkdir ./data/dataset_10000p')
os.system('mkdir ./data/dataset_10000p/3_5')
os.system('mkdir ./data/dataset_10000p/3_5/train')
os.system('mkdir ./data/dataset_10000p/3_5/val')
os.system('mkdir ./data/dataset_10000p/3_5/test')
os.system('mkdir ./data/dataset_10000p/7_10')
os.system('mkdir ./data/dataset_10000p/7_10/train')
os.system('mkdir ./data/dataset_10000p/7_10/val')
os.system('mkdir ./data/dataset_10000p/7_10/test')
os.system('mkdir ./data/dataset_10000p/12_15')
os.system('mkdir ./data/dataset_10000p/12_15/train')
os.system('mkdir ./data/dataset_10000p/12_15/val')
os.system('mkdir ./data/dataset_10000p/12_15/test')

for n_pairs in ["little_dataset_100p/", "dataset_10000p/"]:
    if n_pairs == "little_dataset_100p/":
        n_pairsn = 100
    if n_pairs == "dataset_10000p/":
        n_pairsn = 100000
    for limit in ["3_5/", "7_10/", "12_15/"]:
        if limit=="3_5/":
            min_nn = 3
            max_nn = 5
        if limit=="7_10/":
            min_nn = 7
            max_nn = 10
        if limit=="12_15/":
            min_nn = 12
            max_nn = 15
        for dos in ["train/", "val/", "test/"]:
            if dos == "val" and n_pairsn == 100000:
                n_pairsn = n_pairsn/10
            if dos == "test" and n_pairsn == 100000:
                n_pairsn = n_pairsn / 10
            if dos == "train":
                if n_pairsn > 100:
                    if n_pairsn != 100000:
                        n_pairsn = 100000
            seed = random.randint(0,100)
            path1 = "./data/"+n_pairs+limit+dos+"dimacs"
            path2 = "./data/" + n_pairs + limit + dos + "pickle"
            os.system('mkdir '+ path1)
            os.system('mkdir ' + path2)
            os.system('python3 create_database_random.py --seed '+str(seed)+' --n_pairs '+str(n_pairsn)+' --dimacs_dir '+path1+' --out_dir '+path2+' --min_n '+str(min_nn)+' --max_n '+str(max_nn))


