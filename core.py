import numpy as np
import cv2 as cv
from pathlib import Path
import pandas as pd
from multiprocessing import Pool


def saveToCsv(folderPath,nameList:list,D:np.ndarray):
    output_filename = 'results.csv'
    
    n = len(nameList)
    # Create a list to store distances and their corresponding pairs
    distance_pairs = []
    for i in range(n):
        for j in range(i):
            if not np.isnan(D[i, j]):
                distance_pairs.append((nameList[i], nameList[j], int(D[i, j])))
    
    # Sort the distance_pairs by distance in descending order
    distance_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Create a DataFrame
    df = pd.DataFrame(distance_pairs, columns=['coin 1', 'coin 2', 'number of matches'])
    
    # Write to CSV
    finalPath = Path(folderPath,output_filename)
    df.to_csv(finalPath, index=False,sep=';')


       
def getMatrixAndNumber(kp1,des1,kp2,des2,ratio,ransacReprojThreshold,maxIters:int):
    

    # flann = cv.FlannBasedMatcher()

    # # Perform matching
    # matches = flann.knnMatch(des1, des2, k=2)
        
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)


    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append([m])
    
    # Estimation de la transformation
    goodArray = np.array(good).ravel()
    src_pts = np.float32( [  kp1[m.queryIdx] for m in goodArray]).reshape(-1,1,2)
    dst_pts = np.float32( [  kp2[m.trainIdx] for m in goodArray]).reshape(-1,1,2)

    
    H, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts,method=cv.RANSAC,ransacReprojThreshold=ransacReprojThreshold,maxIters=maxIters)
    nbMatchedFeatures = np.sum(rigid_mask==1)

    #del flann

    return H,nbMatchedFeatures
    


def getSliderImg(img1:np.ndarray,img2:np.ndarray,H:np.ndarray,x:float,l:int)->np.ndarray:
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
    img1Warped = cv.warpAffine(img1, H, (img1.shape[1],img1.shape[0]),borderMode=cv.BORDER_CONSTANT,borderValue=(255,255,255)) 
    cY,cX = _getCenter(img2)
    
    # composition des deux images
    img3 = _getImageCompose(img1Warped,img2,x,l,cX)
    img3RBG = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
    
    #Zoom
    zoomED = img3RBG[cY-l:cY+l,cX-l:cX+l,:]
    return zoomED


def getImgDrawMatchv2(path1,
                      path2,
                        nFeatures:int,
                        contrastThreshold:float,
                        edgeThreshold :float,
                        siftSigma : float,
                        enablePreciseUpscale : bool,
                        nOctaveLayers: int,
                        ratio:float,
                        ransacReprojThreshold:float,
                        maxIters :int,
                        usePreprocessing:bool,
                        preprocessingParam:dict):
    
    clipLimit = preprocessingParam["clipLimit"]
    gridSize = preprocessingParam["gridSize"]
    h = preprocessingParam["h"]

    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(gridSize,gridSize))
    
    sift = cv.SIFT_create(nfeatures=nFeatures,
                        nOctaveLayers=nOctaveLayers,    # Number of layers in each octave
                        contrastThreshold=contrastThreshold,    # Threshold to filter out weak features
                        edgeThreshold=edgeThreshold,    # Threshold for edge rejection
                        sigma=siftSigma,    # Standard deviation for Gaussian smoothing
                        enable_precise_upscale  = enablePreciseUpscale
            )
    
    img1 = cv.imread(str(path1),cv.IMREAD_GRAYSCALE) # queryImage     
    img2 = cv.imread(str(path2),cv.IMREAD_GRAYSCALE) # queryImage    
    # Get the height and width of the image
    height, _ = img1.shape[:2]

    # Remove the bottom 100 pixels
    new_height = height - 100
    img1 = img1[:new_height, :]
    img2 = img2[:new_height, :]


    if usePreprocessing:
        imgHist = clahe.apply(img1)
        img1 =  cv.fastNlMeansDenoising(imgHist,None,h)

        imgHist = clahe.apply(img2)
        img2 =  cv.fastNlMeansDenoising(imgHist,None,h)

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append([m])
    
    # Estimation de la transformation
    goodArray = np.array(good).ravel()
    src_pts = np.float32( [  kp1[m.queryIdx].pt for m in goodArray]).reshape(-1,1,2)
    dst_pts = np.float32( [  kp2[m.trainIdx].pt for m in goodArray]).reshape(-1,1,2)
    
    H, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts,method=cv.RANSAC,ransacReprojThreshold=ransacReprojThreshold,maxIters=maxIters)
    
    goodFiltered = [ match for idx,match in enumerate(good) if rigid_mask.ravel()[idx] == 1 ]

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,goodFiltered,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    nbFeatures = np.sum(rigid_mask==1)
    
    return img3,nbFeatures


def getMatrixFromFolder(folderPath:Path,
                        nFeatures:int,
                        contrastThreshold:float,
                        edgeThreshold :float,
                        siftSigma : float,
                        enablePreciseUpscale : bool,
                        nOctaveLayers: int,
                        ratio:float,
                        ransacReprojThreshold:float,
                        maxIters:int,
                        callback:callable,
                        usePreprocessing:bool,
                        discradLinkOnScale :float,
                        preprocessingParam:dict,
                        numProcessors:int):
    """Calcule la matrice des correspondances D[i,j] = nbFeatures(i,j) et renvoie un tableau contenant
    les noms des fichiers, la matrice de correspondance et une matrice contenant les matrices de transformation

    Args:
        folderPath (Path): Chemin du dossier
        contrastThreshold (float): seuil de contraste (SIFT)
        ratio (float):ratio pour les bonnes correspondances (SIFT)
        callback (callable, optional): fonction qui est appelée avec comme argument le pourcentage d'avancée. Defaults to lambdax:None.

    Returns:
        tuple[list,np.ndarray,np.ndarray]: _description_
    """
    allPath = sorted(list(folderPath.glob("*.jpg")))
    N = len(allPath)

    kpDesList = []

    clipLimit = preprocessingParam["clipLimit"]
    gridSize = preprocessingParam["gridSize"]
    h = preprocessingParam["h"]

    
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(gridSize,gridSize))


    sift = cv.SIFT_create(nfeatures=nFeatures,
                            nOctaveLayers=nOctaveLayers,    # Number of layers in each octave
                            contrastThreshold=contrastThreshold,    # Threshold to filter out weak features
                            edgeThreshold=edgeThreshold,    # Threshold for edge rejection
                            sigma=siftSigma,    # Standard deviation for Gaussian smoothing
                            enable_precise_upscale  = enablePreciseUpscale
                )


    for id in range(N):
        path = str(allPath[id])

        img = cv.imread(path,cv.IMREAD_GRAYSCALE) # queryImage     
        # Get the height and width of the image
        height, _ = img.shape[:2]

        # Remove the bottom 100 pixels
        new_height = height - 100
        img = img[:new_height, :]


        if usePreprocessing:
            imgHist = clahe.apply(img)
            img =  cv.fastNlMeansDenoising(imgHist,None,h)


        kp1, des1 = sift.detectAndCompute(img,None)
        onlyPoints = [kp.pt for kp in kp1]

        kpDesList.append([onlyPoints,des1])

        print(f"preprocessing {id}/{N}",end='\r')

    nKeyPoints = np.array([len(kpDes[1]) for kpDes in kpDesList])
    print(f"nombre de moyen de kp par image : {np.mean(nKeyPoints)}")
    
    print("preprocessing DONE")

    asincResults = []
    
    c= 0

    def resultCallback(res,idRow):
        nonlocal c  # Declare c as a global variable
        c = c+ idRow
        print(c/total*100)
        asincResults.append((res,idRow))
    
    ## Calcul 

    total = N*(N-1)/2

    args = [ (idx ,kpDesList,ratio, ransacReprojThreshold,maxIters) for idx in range(1,N)]
    with Pool(processes=numProcessors) as pool:
        for arg in args:
            pool.apply_async(getRowOfDistance, arg, callback=lambda res, idRow=arg[0]: resultCallback(res, idRow))

        pool.close()
        pool.join()

    # tri du calcul asynchrone
    asincResults.sort(key=lambda x: x[1])
    DAndHInLines = [result[0] for result in asincResults]

    ## reconstruction de la matrice
    DLines = [resIter[0] for resIter in DAndHInLines]    
    print(DLines)
    D = np.empty((N,N))
    D[:] = np.NaN
    for idx1 in range(1,N):
        for idx2 in range(idx1):
                D[idx1,idx2] = DLines[idx1-1][idx2]

    
    HLines = [resIter[1] for resIter in DAndHInLines]
    Hm = np.zeros((N,N,2,3))
    for idx1 in range(1,N):
        for idx2 in range(idx1):
                Hm[idx1,idx2] = HLines[idx1-1][idx2]
    
            
    nameList = [path.name for path in allPath]
            
    return nameList,D,Hm


def getRowOfDistance(idx1,kpDesList,ratio,ransacReprojThreshold,maxIters):
    # calcule une seule ligne (utilisé pour multiprocessing)
    # renvoie une seule ligne de D et une seule ligne de H dans un tuple
    # return (D,H)

    D = []
    Hm = []
    kp1,des1 = kpDesList[idx1]
    for idx2 in range(idx1):
        kp2,des2 = kpDesList[idx2]
    
        H,nbFeatures = getMatrixAndNumber(kp1,des1,kp2,des2,ratio=ratio,ransacReprojThreshold=ransacReprojThreshold,maxIters=maxIters)
        
        s = np.sqrt(H[0,0]**2 +H[1,0]**2)
        # if the scale is too important discard the link
        if np.abs(s-1)>0.25:
            D.append(0)
        else:
            D.append(nbFeatures)

        Hm.append(H)

    return (D,Hm)

def getOrderedLinks(D,Hm):
    
    dim = np.shape(D)
    N,_ =dim 
    Dravel = np.ravel(D)
    nbComparaison = int(N*(N-1)/2)

    ordered = np.dstack(np.unravel_index(np.argsort(-Dravel),dim))[0][:nbComparaison]

    # Si on veut afficher toutes les liaisons : 
    orderedLinks = [link for link in ordered]
    return np.array(orderedLinks)

def isDataAvailable(folderPath):
    dataNameList = [file.name for file in folderPath.glob("*.npy")]
    for file in ["D.npy","Hm.npy","nameList.npy"]:
        if not file in dataNameList:
            return False
    return True

def getSavedData(folderPath):
    nameList = np.load(Path(folderPath,"nameList.npy"))
    D = np.load(Path(folderPath,"D.npy"))
    Hm = np.load(Path(folderPath,"Hm.npy"))
    
    return nameList, D ,Hm 

###### Fonctions auxiliaires ######
#######################################

def _getImageCompose(img1,img2,x,l,cX):
    ## doivent avoir la même taille
    if np.shape(img1) != np.shape(img2):
        raise Exception("Les deux images doivent avoir la même taille")
    #taille de l'image 2l
    
    nbPixel = int(2*l*x)
    lim = cX - l + nbPixel
    return cv.hconcat([img1[:,:lim],img2[:,lim:]])

def _getCenter(img):
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img,(5,5),0)
    ret,th = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    center = [ np.average(indices) for indices in np.where(th >= 255) ]
    
    cY,cX = int(center[0]),int(center[1])
    return cY,cX





