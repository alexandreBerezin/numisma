#### exemple de script pour lancer le calcul #### 
from pathlib import Path
import numpy as np

from core import KeypointDetector,imagesMatcher,saveToCsv


## fonction pour afficher la progression
def callback_detector(current:int,total:str)->None:
    print(f" {current}/{total}")

def callback_matcher(progress:int,time:str)->None:
    print(f" {progress:.2f}% temps restant : {time}")


## clause pour éviter de lancer récursivement une chaine de processus
if __name__ == "__main__":

    # chemin du dossier avec les images
    path = "./numisma/data/dataTest1"

    ## prétraitement des données
    print("preprocessing ...")
    detector = KeypointDetector(folder_path=path,callback=callback_detector) ## avec les paramètres de base
    kp_and_des,list_of_names = detector.get_kp_des_from_folder()


    print("start matching ...")
    matcher = imagesMatcher(kp_and_des,callback=callback_matcher,n_processes=1) # avec les paramètres de base


    # lancement du calcul
    N,Hm = matcher.get_N_and_H()


    print("enregistrement des données")
    # enregistrement des resultats
    np.save(Path(path,"N.npy"),N)
    np.save(Path(path,"Hm.npy"),Hm)
    np.save(Path(path) / "detector_param.npy",detector.param,allow_pickle=True)
    np.save(Path(path) / "matcher_param.npy",matcher.param,allow_pickle=True)


    #enregistrement csv
    saveToCsv(path,detector.all_path,N)

    

    print("fin du calcul")