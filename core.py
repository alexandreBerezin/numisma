from pathlib import Path
from multiprocessing import Pool
import threading
import time
import os
import shutil
import datetime
from collections.abc import Callable
from typing import List, Tuple


import pandas as pd
import numpy as np
import cv2 as cv



class KeypointDetector():
    """Classe utilisé pour la detection et la desctiption des points d'intérêts
    """
    def __init__(self,
                 folder_path: os.PathLike,
                 use_preprocessing: bool = True,
                 clip_limit: float = 0.002,
                 grid_size: int = 10,
                 h: float = 20,
                 n_octave_layers: int = 7,
                 contrast_threshold: float = 0.02,
                 edge_treshold: float = 100,
                 sift_sigma: float = 1.1,
                 callback: Callable[[int, int], None] = None) -> None:

        self.folder_path = folder_path


        list_path = list(Path(self.folder_path).glob("*.jpg"))
        if not list_path:
            raise Exception("Pas d'image .jpg dans le dossier")

        # pour la reproductibilité entre différents os
        self.all_path = sorted(list_path)


        self.param = {
            "use_preprocessing":use_preprocessing,
            "clip_limit" : clip_limit,
            "grid_size":grid_size,
            "h":h,
            "n_octave_layers":n_octave_layers,
            "contrast_threshold": contrast_threshold,
            "edge_treshold":edge_treshold,
            "sift_sigma":sift_sigma,
        }

        ###### Preprocessing param ########
        self.use_preprocessing = use_preprocessing
        self.h = h
        self.callback = callback

        if use_preprocessing:
            # Si on untilise le prétraitement des données
            # CLAHE : Contrast limited Adaptative Histogram Equalization
            self.clahe = cv.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

        ####### Keypoint detection params #######
        self.sift = cv.SIFT_create(nfeatures=0,
                                   nOctaveLayers=n_octave_layers,
                                   contrastThreshold=contrast_threshold,
                                   edgeThreshold=edge_treshold,
                                   sigma=sift_sigma,
                                   )

    def get_img_from_path(self,path_image:str):
        img = cv.imread(path_image, cv.IMREAD_GRAYSCALE)

        height, _ = img.shape[:2]

        # On enlève la partie basse des images
        new_height = height - 100
        img = img[:new_height, :]

        return img
    
    def get_kp_desc_from_image(self, path: str) -> Tuple[list, np.ndarray]:
        img = self.get_img_from_path(path)

        if self.use_preprocessing:
            img_hist = self.clahe.apply(img)
            img = cv.fastNlMeansDenoising(img_hist, None, self.h)

        kp1, des = self.sift.detectAndCompute(img, None)
        only_points = [kp.pt for kp in kp1]

        return only_points, des

    def get_kp_des_from_folder(self) -> Tuple[list, list]:
        # get all path of images

        N = len(self.all_path)

        kp_des_list = []

        for id in range(N):
            path = str(self.all_path[id])
            pt, des = self.get_kp_desc_from_image(path)
            kp_des_list.append([pt, des])

            # callback pour afficher la progression
            if self.callback is not None:
                self.callback(id, N)


        self.kp_des_list = kp_des_list

        return kp_des_list, self.all_path


class imagesMatcher():

    def __init__(self,
                 kp_des_list: List[list]=None,
                 N : np.ndarray=None,
                 Hm : np.ndarray=None,
                 n_processes: int = 1,
                 ratio: float = 0.99,
                 ransac_reproj_treshold: float = 3,
                 max_iter_ransac: int = 10_000_000,
                 ransac_confidence: float = 0.95,
                 callback: Callable[[int, str], None] = None) -> None:
        

        # vérifie que la classe est bien initialisée
        # soit par des kp soit par les matrices
        if (N is None and Hm is None) and kp_des_list is None:
            # erreur d'initialisation
            raise Exception("Initialiser ImageMatcher avec la liste de kp ou avec les matrices N et Hm")



        self.kp_des_list = kp_des_list
        self.n_processes = n_processes
        self.ratio = ratio
        self.ransac_reproj_treshold = ransac_reproj_treshold
        self.max_iter_ransac = max_iter_ransac
        self.callback = callback
        self.ransac_confidence = ransac_confidence

        if kp_des_list is not None : self.nb_images = len(kp_des_list)
        else: self.nb_images,_ = np.shape(N)

        # dict param to save
        self.param = {
            "ratio":ratio,
            "ransac_reproj_treshold":ransac_reproj_treshold,
            "max_iter_ransac":max_iter_ransac,
            "ransac_confidence":ransac_confidence,
        }
        
        self.Hm =Hm
        self.N = N

        self.sub_folder_path = Path.cwd() / 'monitoring'

    @staticmethod
    def _get_row_matches(idx: int,
                         kp_des_list: List[list],
                         ratio: float, ransac_reproj_treshold: float,
                         max_iter_ransac: int,
                         ransac_confidence: float,
                         sub_folder_path: os.PathLike) -> Tuple[list, list]:
        """Une fonction qui calculer une seule ligne (idx) des matrices de distance 
        et des matrices de projection
        """
        D = []
        Hm = []

        # Brute force matcher
        bf = cv.BFMatcher()

        kp1, des1 = kp_des_list[idx]
        for idx2 in range(idx):

            kp2, des2 = kp_des_list[idx2]
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < ratio*n.distance:
                    good.append([m])

            # Estimation de la transformation
            goodArray = np.array(good).ravel()
            src_pts = np.float32([kp1[m.queryIdx]
                                 for m in goodArray]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx]
                                 for m in goodArray]).reshape(-1, 1, 2)

            H, rigid_mask = cv.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv.RANSAC,
                ransacReprojThreshold=ransac_reproj_treshold,
                maxIters=max_iter_ransac,
                confidence=ransac_confidence)

            n_matched_features = np.sum(rigid_mask == 1)
            Hm.append(H)

            s = np.sqrt(H[0,0]**2 +H[1,0]**2)
            if np.abs(s-1)>0.25:
                D.append(0)
            else:
                D.append(n_matched_features)

        # lorsque chaque processus a fini de travailler, il va créer un fichier
        # texte avec l'id de la ligne
        file_path = os.path.join(sub_folder_path, f"process_{idx}.txt")
        with open(file_path, 'w') as file:
            pass  # write an empty file

        return (D, Hm)

    def observateur(self):
        total = self.nb_images*(self.nb_images-1)/2
        last_progress = 0
        t0 = time.time()
        if not os.path.exists(self.sub_folder_path):
            os.makedirs(self.sub_folder_path)

        while not self.exit_event.is_set():
            # créer le dossier si il n'existe pas

            files_in_folder = os.listdir(self.sub_folder_path)
            sum_of_process_ids = 0

            # on fait la somme de tous les ids de ligne
            for file_name in files_in_folder:
                if file_name.startswith("process_") and file_name.endswith(".txt"):
                    process_id = int(file_name.split("_")[1].split(".")[0])
                    sum_of_process_ids += process_id

            # on calcule la progression totale
            progress = sum_of_process_ids/total
            if progress != 0:
                dtOveravancement = ((time.time()-t0))/progress
            else:
                dtOveravancement = 0

            if progress != last_progress:
                progression = progress*100
                remainingTime = str(datetime.timedelta(
                    seconds=int((1-progress)*dtOveravancement)))
                if self.callback is not None:
                    self.callback(progression, remainingTime)
                last_progress = progress
            time.sleep(5)

    def get_N_and_H(self,):

        # Event utilisé pour sortir du thread observateur
        self.exit_event = threading.Event()

        # Lancement de l'observateur
        observateur = threading.Thread(target=self.observateur, args=())
        observateur.start()

        # arguments à passer aux fonctions
        args = [(idx, self.kp_des_list, self.ratio, self.ransac_reproj_treshold, self.max_iter_ransac, self.ransac_confidence, self.sub_folder_path)
                for idx in range(1, self.nb_images)]

        # multiprocessing par ligne des matrices N et H
        func = imagesMatcher._get_row_matches
        with Pool(processes=self.n_processes) as pool:
            list_of_line_of_D_and_H = pool.starmap(func, args)

        # To close the thread from the main thread
        self.exit_event.set()
        observateur.join()

        shutil.rmtree(self.sub_folder_path)

        # reconstruction de la matrice
        rows_of_D = [resIter[0] for resIter in list_of_line_of_D_and_H]

        N = np.empty((self.nb_images, self.nb_images))
        N[:] = np.NaN
        for idx1 in range(1, self.nb_images):
            for idx2 in range(idx1):
                N[idx1, idx2] = rows_of_D[idx1-1][idx2]


        rows_of_H = [resIter[1] for resIter in list_of_line_of_D_and_H]
        Hm = np.zeros((self.nb_images, self.nb_images, 2, 3))
        for idx1 in range(1, self.nb_images):
            for idx2 in range(idx1):
                Hm[idx1, idx2] = rows_of_H[idx1-1][idx2]

        self.N = N
        self.Hm = Hm

        return N, Hm

    def get_matching_mask(self, kp1: list, des1: np.ndarray, kp2: list, des2: np.ndarray):
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < self.ratio*n.distance:
                good.append([m])

        # Estimation de la transformation
        goodArray = np.array(good).ravel()
        src_pts = np.float32([kp1[m.queryIdx]
                             for m in goodArray]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx]
                             for m in goodArray]).reshape(-1, 1, 2)

        H, rigid_mask = cv.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_treshold,
            maxIters=self.max_iter_ransac,
            confidence=self.ransac_confidence)

        matching_mask = [match for idx, match in enumerate(
            good) if rigid_mask.ravel()[idx] == 1]

        return matching_mask


def get_draw_match_image_by_id(id1: int,
                         id2: int,
                         key_point_detector: KeypointDetector,
                         image_matcher: imagesMatcher) -> Tuple[np.ndarray, int]:


    
    path1 = key_point_detector.all_path[id1]
    path2 = key_point_detector.all_path[id2]

    img1 = key_point_detector.get_img_from_path(str(path1))
    img2 = key_point_detector.get_img_from_path(str(path2))

    kp1, des1 = key_point_detector.get_kp_desc_from_image(str(path1))
    kp2, des2 = key_point_detector.get_kp_desc_from_image(str(path2))

    opencv_kp1 = [cv.KeyPoint(x, y, 1) for x, y in kp1]
    opencv_kp2 = [cv.KeyPoint(x, y, 1) for x, y in kp2]

    matching_mask = image_matcher.get_matching_mask(kp1, des1, kp2, des2)


    img3 = cv.drawMatchesKnn(img1, opencv_kp1, img2, opencv_kp2, matching_mask,
                             None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    nbFeatures = np.sum(matching_mask == 1)

    return img3, nbFeatures


def get_draw_match_image_by_rank(rank: int,
                                 key_point_detector: KeypointDetector,
                                 image_matcher: imagesMatcher) -> Tuple[np.ndarray, int]:

    all_path = key_point_detector.all_path
    folder_path = key_point_detector.folder_path
    
    N = image_matcher.N

    
    #get the number
    N_copy = np.copy(N)
    # Assuming 'matrix' is your original matrix
    sorted_indices = np.argsort(-np.nan_to_num(N_copy, nan=-np.inf), axis=None)
        

    #sorted_values = np.nan_to_num(D_copy, nan=np.inf).ravel()[sorted_indices]
    sorted_indices = np.unravel_index(sorted_indices, N_copy.shape)
    

    list_id1,list_id2 = sorted_indices
    id1 = list_id1[rank]
    id2 = list_id2[rank]
    

    path1 =all_path[id1]
    path2 = all_path[id2]
    
    print(path1.name)
    print(path2.name)
    
    kp1, des1 = key_point_detector.get_kp_desc_from_image(str(path1))
    kp2, des2 = key_point_detector.get_kp_desc_from_image(str(path2))

    opencv_kp1 = [cv.KeyPoint(x, y, 1) for x, y in kp1]
    opencv_kp2 = [cv.KeyPoint(x, y, 1) for x, y in kp2]

    matching_mask = image_matcher.get_matching_mask(kp1, des1, kp2, des2)

    img1 = key_point_detector.get_img_from_path(str(path1))
    img2 = key_point_detector.get_img_from_path(str(path2))

    img3 = cv.drawMatchesKnn(img1, opencv_kp1, img2, opencv_kp2, matching_mask,
                             None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    
    
    nbFeatures = np.sum(matching_mask == 1)

    return img3, len(matching_mask)


def saveToCsv(folderPath, nameList: list, D: np.ndarray):
    output_filename = 'results.csv'

    n = len(nameList)
    # Create a list to store distances and their corresponding pairs
    distance_pairs = []
    for i in range(n):
        for j in range(i):
            if not np.isnan(D[i, j]):
                distance_pairs.append((nameList[i].name, nameList[j].name, int(D[i, j])))

    # Sort the distance_pairs by distance in descending order
    distance_pairs.sort(key=lambda x: x[2], reverse=True)

    # Create a DataFrame
    df = pd.DataFrame(distance_pairs, columns=[
                      'coin 1', 'coin 2', 'number of matches'])

    # Write to CSV
    finalPath = Path(folderPath, output_filename)
    df.to_csv(finalPath, index=False, sep=';')


def getSliderImg(img1: np.ndarray, img2: np.ndarray, H: np.ndarray, x: float, l: int) -> np.ndarray:
    """Crée une image qui est composé en partie de l'image 1 transformée par la matrice H et de l'image 2

    Args:
        img1 (np.ndarray): image1 OpenCV BGR
        img2 (np.ndarray): image2 OpenCV BGR
        H (np.ndarray): Matrice de transformation rigide
        x (float): position de la coupure entre les deux images 0<x<1
        l (int): largeur en pixel de l'image centrée sur la zone d'importance

    Returns:
        np.ndarray: image Composite matplotlib RGB 
    """

    # transformation de l'image 1
    img1Warped = cv.warpAffine(
        img1, H, (img1.shape[1], img1.shape[0]), borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))
    cY, cX = _getCenter(img2)


    # composition des deux images
    img3 = _getImageCompose(img1Warped, img2, x, l, cX)
    img3RBG = cv.cvtColor(img3, cv.COLOR_BGR2RGB)


    # Zoom
    zoomED = img3RBG[cY-l:cY+l, cX-l:cX+l, :]
    return zoomED


def getOrderedLinks(D):

    dim = np.shape(D)
    N, _ = dim
    Dravel = np.ravel(D)
    nbComparaison = int(N*(N-1)/2)

    ordered = np.dstack(np.unravel_index(
        np.argsort(-Dravel), dim))[0][:nbComparaison]

    # Si on veut afficher toutes les liaisons :
    orderedLinks = [link for link in ordered]
    return np.array(orderedLinks)


def is_data_available(folderPath:os.PathLike):
    dataNameList = [file.name for file in Path(folderPath).glob("*.npy")]
    for file in ["N.npy", "Hm.npy","detector_param.npy","matcher_param.npy"]:
        if not file in dataNameList:
            return False
    return True


def get_saved_data(folderPath):

    N = np.load(Path(folderPath, "N.npy"))
    Hm = np.load(Path(folderPath, "Hm.npy"))

    detector_param = np.load(Path(folderPath) / "detector_param.npy",allow_pickle=True).tolist()
    matcher_param = np.load(Path(folderPath) / "matcher_param.npy",allow_pickle=True).tolist()



    return N, Hm,detector_param,matcher_param

###### Fonctions auxiliaires ######
#######################################


def _getImageCompose(img1, img2, x, l, cX):
    # doivent avoir la même taille
    if np.shape(img1) != np.shape(img2):
        raise Exception("Les deux images doivent avoir la même taille")
    # taille de l'image 2l

    nbPixel = int(2*l*x)
    lim = cX - l + nbPixel
    return cv.hconcat([img1[:, :lim], img2[:, lim:]])


def _getCenter(img):

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    center = [np.average(indices) for indices in np.where(th >= 255)]

    cY, cX = int(center[0]), int(center[1])
    return cY, cX
