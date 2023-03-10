import numpy as np
import cv2 as cv
from pathlib import Path
import time

def getImgDrawMatch(path1:Path,path2:Path,ratio:float=0.8,contrastThreshold:float=0.02,ransacReprojThreshold:float=10)->np.ndarray:
    """renvoie une image qui montre les points de correspondance entre les deux images. 
    Utile pour tester l'algorithme.

    Args:
        path1 (Path): chemin de l'image 1
        path2 (Path): chemin de l'image 2 

    Returns:
        np.ndarray: image contenant les correspondances
    """

    # chargement des images
    t = time.time()
    img1 = cv.imread(str(path1),cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(str(path2),cv.IMREAD_GRAYSCALE) # trainImage
    print(f"chargement images : {time.time()-t}")
    
    # Initiate SIFT detector
    t = time.time()
    sift = cv.SIFT_create(contrastThreshold=contrastThreshold)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print(f"SIFT detect : {time.time()-t}")
    
    
    # BFMatcher with default params
    t = time.time()
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    print(f"BF matcher : {time.time()-t}")
    
    # Apply ratio test
    t = time.time()
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append([m])

    goodArray = np.array(good).ravel()
    src_pts = np.float32( [  kp1[m.queryIdx].pt for m in goodArray]).reshape(-1,1,2)
    dst_pts = np.float32( [  kp2[m.trainIdx].pt for m in goodArray]).reshape(-1,1,2)
    print(f"Ratio test : {time.time()-t}")
    
    # rigid estimation 
    t = time.time()
    H, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts,method=cv.RANSAC,ransacReprojThreshold=ransacReprojThreshold)
    print(f"Rigid estimation : {time.time()-t}")
    goodFiltered = [ match for idx,match in enumerate(good) if rigid_mask.ravel()[idx] == 1 ]
    


    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,goodFiltered,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    nbFeatures = np.sum(rigid_mask==1)
    
    return img3,nbFeatures


def getKpDes(img,contrastThreshold):
    sift = cv.SIFT_create(contrastThreshold=contrastThreshold)
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img,None)
    return kp,des
    
    
def getMatrixAndNumber(kp1,des1,kp2,des2,ratio):
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
    
    H, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts,method=cv.RANSAC,ransacReprojThreshold=10)
    nbMatchedFeatures = np.sum(rigid_mask==1)

    return H,nbMatchedFeatures
    

def getMatchedFeaturesNumber(img1:np.ndarray,img2:np.ndarray,contrastThreshold:float,ratio:float)->tuple[np.ndarray,int]:
    """Fonction qui renvoie la matrice de transformation rigide (2*3) et le
    nombre de correspondances entre les deux images
    
    La fonction utilise la fonction estimateAffinePartial2D() de OpenCV en particulier la 
    methode RANSAC pour ??liminer les correspondances qui ne sont pas pertinantes au vue d'une
    transformation rigide
    
    Args:
        img1 (np.ndarray): image1 OpenCV RBG
        img2 (np.ndarray): image1 OpenCV RBG
        contrastThreshold (float): seuil de contraste (SIFT)
        ratio (float):ratio pour les bonnes correspondances (SIFT)

    Returns:
        Tuple[np.ndarray,int]: (H,nbMatchedFeatures) 
    """
    
    # Initiate SIFT detector
    sift = cv.SIFT_create(contrastThreshold=contrastThreshold)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
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
    
    H, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts,method=cv.RANSAC,ransacReprojThreshold=10)
    nbMatchedFeatures = np.sum(rigid_mask==1)

    return H,nbMatchedFeatures

def getSliderImg(img1:np.ndarray,img2:np.ndarray,H:np.ndarray,x:float,l:int)->np.ndarray:
    """Cr??e une image qui est compos?? en partie de l'image 1 transform??e par la matrice H et de l'image 2

    Args:
        img1 (np.ndarray): image1 OpenCV BGR
        img2 (np.ndarray): image2 OpenCV BGR
        H (np.ndarray): Matrice de transformation rigide
        x (float): position de la coupure entre les deux images 0<x<1
        l (int): largeur en pixel de l'image centr??e sur la zone d'importance

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

def getMatrixFromFolder(folderPath:Path,contrastThreshold:float,ratio:float,callback:callable=lambda x:None)->tuple[list,np.ndarray,np.ndarray]:
    """Calcule la matrice des correspondances D[i,j] = nbFeatures(i,j) et renvoie un tableau contenant
    les noms des fichiers, la matrice de correspondance et une matrice contenant les matrices de transformation

    Args:
        folderPath (Path): Chemin du dossier
        contrastThreshold (float): seuil de contraste (SIFT)
        ratio (float):ratio pour les bonnes correspondances (SIFT)
        callback (callable, optional): fonction qui est appel??e avec comme argument le pourcentage d'avanc??e. Defaults to lambdax:None.

    Returns:
        tuple[list,np.ndarray,np.ndarray]: _description_
    """
    allPath = sorted(list(folderPath.glob("*.jpg")))
    N = len(allPath)
    D = np.empty((N,N))
    D[:] = np.NaN
    
    Hm = np.zeros((N,N,2,3))
    c=0
    
    total = (N-1)*N/2
    
    kpDesList = []
    
    for idx1 in range(N): 
        img1 = cv.imread(str(allPath[idx1]),cv.IMREAD_GRAYSCALE) # queryImage     
        kp1,des1 = getKpDes(img=img1,contrastThreshold=contrastThreshold)
        kpDesList.append([kp1,des1])
        
        for idx2 in range(idx1):
            kp2,des2 = kpDesList[idx2]
            c=c+1
        
            H,nbFeatures = getMatrixAndNumber(kp1,des1,kp2,des2,ratio=ratio)
            
            D[idx1,idx2]=nbFeatures
            Hm[idx1,idx2]=H
            
            pourcent = 100*c/total

            print(f":{pourcent:.2f}%         ",end='\r')
            callback(c,total)
            
    nameList = [path.name for path in allPath]
            
    return nameList,D,Hm

def getOrderedLinks(D,Hm):
    
    dim = np.shape(D)
    N,_ =dim 
    Dravel = np.ravel(D)
    nbComparaison = int(N*(N-1)/2)

    ordered = np.dstack(np.unravel_index(np.argsort(-Dravel),dim))[0][:nbComparaison]
    orderedLinks = [link for link in ordered if _filterLink(link,Hm)]
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
    ## doivent avoir la m??me taille
    if np.shape(img1) != np.shape(img2):
        raise Exception("Les deux images doivent avoir la m??me taille")
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

def _filterLink(link,Hm):
    id1,id2 = link
    H = Hm[id1,id2]
    s = H[0,0]**2 +H[1,0]**2
    return np.abs(s-1) < 0.5





