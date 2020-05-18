import os
from tqdm import tqdm

filenames = os.listdir("./data/Mini_GIFT_poubelle")

dico_name = {"5":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}, "6":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}}

for filename in tqdm(filenames):
    liste_interet = filename.split('_')
    dico_name[liste_interet[0]][liste_interet[-1]].append(int(liste_interet[1]))

for round in ["5", '6']:
    lst1 = dico_name[round]["UNSAT.DIMACS"]
    lst2 = dico_name[round]["SAT.DIMACS"]
    lst3 = [value for value in lst1 if value in lst2]
    print(max(lst3), round)