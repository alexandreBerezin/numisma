##### CALCUL #######
#Si 1 utilise le prétraitement des données
USE_PREPROCESSING = 1
#utilisé si usePreprocessing = 1 
# voir notebook pour plus d'info sur les différents paramètres
CLIP_LIMIT = 2
GRID_SIZE = 10
PREPROCESSING_H = 22


#utilisé pour SIFT
N_FEATURES = 0
CONTRAST_THRESHOLD = 0.05
EDGE_THRESHOLD  = 40
SIFT_SIGMA = 1.2
ENABLE_PRECISE_UPSCALE = 1
N_OCTAVE_LAYERS = 8

# supprime les liaisons qui ont une transformation aberrante
# plus la valeur est petite plus on filtre les liaisons
DISCARD_LINK_ON_SCALE = 0.25

## Correspondance points d'interêts
RANSAC_REPROJ_THRESHOLD = 4
RATIO = 0.9
MAX_ITER = 5000



######Controls ######
SLIDER_SPEED=3

###### Graphics ######
ZOOM = 250
ZOOM2 = 500
# si 1, filtre les liaisons abec les valeurs aberrantes
USE_FILTER = 0
