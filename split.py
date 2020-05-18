import os
from tqdm import tqdm

PATH = "./data/Mini_GIFT_poubelle" #"./data/mini_GIFT"
#filenames = os.listdir("./data/Mini_GIFT_poubelle")
filenames = os.listdir(PATH)

dico_name = {"5":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}, "6":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}, "7":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}}

for filename in tqdm(filenames):
    liste_interet = filename.split('_')
    dico_name[liste_interet[0]][liste_interet[-1]].append(int(liste_interet[1]))

dico_max = {"5":0, "6":0}
cpt = 0

for round in ["5", '6']:
    lst1 = dico_name[round]["UNSAT.DIMACS"]
    lst2 = dico_name[round]["SAT.DIMACS"]
    lst3 = [max(lst1), max(lst2)]
    dico_max[round] = max(lst3)

for filename in tqdm(filenames):
    liste_interet = filename.split('_')
    if liste_interet[0] == "7":
        os.remove(PATH + "/"+ filename)
        cpt +=1
    else:
        if int(liste_interet[1])>dico_max[liste_interet[0]]:
            os.remove(PATH +"/"+ filename)
            cpt +=1

print(cpt)

