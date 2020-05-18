import os
from tqdm import tqdm

PATH = "./data/mini_GIFT"
#"./data/Mini_GIFT_poubelle" #"./data/mini_GIFT"
#filenames = os.listdir("./data/Mini_GIFT_poubelle")
filenames = os.listdir(PATH)

dico_name = {"5":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}, "6":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}, "7":{"UNSAT.DIMACS":[], "SAT.DIMACS":[]}}

for filename in tqdm(filenames):
    if ("dico_" in filename ):
        liste_interet = filename.split('_')
        dico_name[liste_interet[0]][liste_interet[-1]].append(int(liste_interet[1]))

dico_max = {"5":0, "6":0, "7":0}
cpt = 0

for round in ["5", '6', "7"]:
    lst1 = dico_name[round]["UNSAT.DIMACS"]
    lst2 = dico_name[round]["SAT.DIMACS"]
    if len(lst1)>0 and len(lst2)>0:
        lst3 = [max(lst1), max(lst2)]
        dico_max[round] = max(lst3)




os.makedirs(PATH + "/train/dimacs/", exist_ok=True)
os.makedirs(PATH + "/train/pickle/", exist_ok=True)
os.makedirs(PATH + "/val/dimacs/", exist_ok=True)
os.makedirs(PATH + "/val/pickle/", exist_ok=True)
os.makedirs(PATH + "/test/dimacs/", exist_ok=True)
os.makedirs(PATH + "/test/pickle/", exist_ok=True)


for round in ["5", '6', "7"]:
    print(round)
    print(dico_max[round]-1)
    start_train = 0
    end_train =  int(0.6*dico_max[round]-1)
    start_val =  int(0.6*dico_max[round]-1)
    end_val =  int(0.8*dico_max[round]-1)
    start_test = int(0.8*dico_max[round]-1)
    end_test =dico_max[round]-1
    for index in tqdm(range(start_train, end_train)):
        for end in ["UNSAT.DIMACS", "SAT.DIMACS"]:
            liste_interet2 = str(round)+"_"+str(index+1)+"_"+liste_interet[2] +"_"+liste_interet[3]+"_" + end
            os.rename(PATH+"/"+ liste_interet2, PATH+ "/train/dimacs/" +liste_interet2)
    for index in tqdm(range(start_val, end_val)):
        for end in ["UNSAT.DIMACS", "SAT.DIMACS"]:
            liste_interet2 = str(round)+"_"+str(index+1)+"_"+liste_interet[2] +"_"+liste_interet[3]+"_" + end
            os.rename(PATH+"/"+ liste_interet2, PATH+ "/val/dimacs/" +liste_interet2)
    for index in tqdm(range(start_test, end_test)):
        for end in ["UNSAT.DIMACS", "SAT.DIMACS"]:
            liste_interet2 = str(round)+"_"+str(index+1)+"_"+liste_interet[2] +"_"+liste_interet[3]+"_" + end
            os.rename(PATH+"/"+ liste_interet2, PATH+ "/test/dimacs/" +liste_interet2)