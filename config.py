import os

##### CALCUL #######
#Si 1 utilise le prétraitement des données
USE_PREPROCESSING = 1
#utilisé si usePreprocessing = 1 
# voir notebook pour plus d'info sur les différents paramètres
CLIP_LIMIT = 3
GRID_SIZE = 3
PREPROCESSING_H = 15


#utilisé pour SIFT
N_FEATURES = 0
CONTRAST_THRESHOLD = 0.06
EDGE_THRESHOLD  = 30
SIFT_SIGMA = 1.1
ENABLE_PRECISE_UPSCALE = 1
N_OCTAVE_LAYERS = 7

# supprime les liaisons qui ont une transformation aberrante
# plus la valeur est petite plus on filtre les liaisons
DISCARD_LINK_ON_SCALE = 0.25

## Correspondance points d'interêts
RANSAC_REPROJ_THRESHOLD = 4
RATIO = 0.95
MAX_ITER = 30000

NUM_PROCESSORS = 2
###CHECKS if environment vriable for slurm 
if os.getenv("SLURM_NTASKS") is not None:
    print(f"overrite number of processes form slurm {os.getenv('SLURM_NTASKS')}")
    NUM_PROCESSORS = int(os.getenv("SLURM_NTASKS"))



######Controls ######
SLIDER_SPEED=3

###### Graphics ######
ZOOM = 250
ZOOM2 = 500

