import os


NUM_PROCESSORS = 2
# CHECKS if environment vriable for slurm
if os.getenv("SLURM_NTASKS") is not None:
    print(
        f"override number of processes form slurm : {os.getenv('SLURM_NTASKS')}")
    NUM_PROCESSORS = int(os.getenv("SLURM_NTASKS"))

DETECTOR_PARAM = {
    'use_preprocessing': True,
    'clip_limit': 0.002,
    'grid_size': 10,
    'h': 20,
    'n_octave_layers': 7,
    'contrast_threshold': 0.02,
    'edge_treshold': 100,
    'sift_sigma': 1.1
}

MATCHER_PARAM = {
    'ratio': 0.99,
    'ransac_reproj_treshold': 6,
    'max_iter_ransac': 10000000,
    'ransac_confidence': 0.99}


###### Controls ######
SLIDER_SPEED = 3

###### Graphics ######
ZOOM = 250
ZOOM2 = 500
