import tkinter as tk   
import tkinter.filedialog as fd  
import tkinter.ttk as ttk  

from pathlib import Path  

import cv2 as cv
from PIL import Image,ImageTk

import threading
import configparser

import time
import numpy as np

### Lecture des paramètres depuis config.ini
config = configparser.ConfigParser()
parentPath = Path(Path(__file__).parent)
import core

config.read(Path(parentPath,"config.ini"))

SLIDER_SPEED = int(config["controls"]["sliderSpeed"])
CONTRAST_THRESHOLD = float(config["calcul"]["contrastThreshold"])
RATIO = float(config["calcul"]["ratio"])
USE_PREPROCESSING = bool(int(config["calcul"]["usePreprocessing"]))

preprocessingParam = {
    "clipLimit" : float(config["calcul"]["clipLimit"]),
    "gridSize": int(config["calcul"]["gridSize"]),
    "h" : float(config["calcul"]["h"]),
}

print(preprocessingParam)


USE_FILTER = bool(int(config["graphics"]["useFilter"]))
ZOOM = int(config["graphics"]["zoom"])
ZOOM2 = int(config["graphics"]["zoom2"])


class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)


        ### top frame
        topFrame = tk.Frame(self,highlightbackground="black", highlightthickness=1)
        topFrame.pack(side="top",fill=tk.X)
        
        titleLabel = tk.Label(master=topFrame,text="Numisma",padx=5,pady=5)
        titleLabel.pack(side="left")
        
        versionLabel = tk.Label(master=topFrame,text="04/02/2023")
        versionLabel.pack(side="right")
        
        
        ### main container
        self.container = tk.Frame(self,width=100)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        

        self.frames = {}
        
        F = StartPage
        page_name = F.__name__
        frame = F(parent=self.container, controller=self)
        self.frames[page_name] = frame

        # put all of the pages in the same location;
        # the one on the top of the stacking order
        # will be the one that is visible.
        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")


    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
        
    def updateParam(self,nameList, D ,Hm,orderedLinks,folderPath):
        print("Update Param")
        self.nameList = nameList
        self.D = D
        self.Hm = Hm
        self.orderedLinks = orderedLinks
        self.folderPath = folderPath
        
        F = ComparePage
        page_name = F.__name__
        frame = F(parent=self.container, controller=self)
        self.frames[page_name] = frame
        # put all of the pages in the same location;
        # the one on the top of the stacking order
        # will be the one that is visible.
        frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame("ComparePage")
        
    def startNewComputation(self,folderPath):
        print("START COMPUTATION")
        F = ComputationPage
        page_name = F.__name__
        frame = F(parent=self.container, controller=self,folderPath=folderPath)
        self.frames[page_name] = frame
        # put all of the pages in the same location;
        # the one on the top of the stacking order
        # will be the one that is visible.
        frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame("ComputationPage")
        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Sélectionner le dossier contenant les images")
        label.pack(side="top", fill="x", pady=30,padx=10)

        button1 = tk.Button(self, text="Dossier",
                            command=lambda: self.selectFolder())

        button1.pack(padx=20,pady=20)
        
        
        
            
    def selectFolder(self):
        dir = fd.askdirectory()
        folderPath = Path(dir)

        if core.isDataAvailable(folderPath):
            
            MsgBox = tk.messagebox.askquestion ('Données de calcul',"Des données d'un calcul précédent ont été trouvées, voulez vous les utiliser ? ")
            if MsgBox == 'yes':
                print("data available")
            
                nameList, D ,Hm = core.getSavedData(folderPath)
                orderedLinks = core.getOrderedLinks(D,Hm,USE_FILTER)
                
                self.controller.updateParam(nameList, D ,Hm,orderedLinks,folderPath)
                
                self.controller.frames["ComparePage"].setNewImg()
                self.controller.frames["ComparePage"].updateFromSlider(50)
                
                self.controller.show_frame("ComparePage")

            else:
                self.controller.startNewComputation(folderPath)
            

            
        else:
            self.controller.startNewComputation(folderPath)
        
class ComparePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        self.controller = controller
        
        self.index = 0 
        self.numberLinks,_ = np.shape(self.controller.orderedLinks)
        
        ###Top folder Path
        
        label = tk.Label(self, text=f"Dossier : {self.controller.folderPath}")
        label.pack(side="top", fill="x", pady=2)
        
        
        self.labelCoin = tk.Label(self, text=f"Monnaie1")
        self.labelCoin.pack(side="top", pady=5,padx=5)
        
        
        ### Label and slider Frame
        ImgAndSliderFrame = tk.Frame(master=self)
        ImgAndSliderFrame.pack()
        
        self.labelImg = tk.Label(master=ImgAndSliderFrame)
        self.labelImg.pack(side="top")
        
        self.slider = tk.Scale(master=ImgAndSliderFrame, from_=0, to=100, orient=tk.HORIZONTAL,command=self.updateFromSlider)
        self.slider.pack(side="top",fill=tk.X)
        
        btn_next = tk.Button(master=ImgAndSliderFrame, text="Suivant",command=self.nextLink)
        btn_next.pack(side="right",padx=10,pady=10)
        
        btn_before = tk.Button(master=ImgAndSliderFrame, text="Précédent",command=self.beforeLink)
        btn_before.pack(side="left",padx=10,pady=10)
        
        btn_mode = tk.Button(master=ImgAndSliderFrame, text="mode",command=self.changeMode)
        btn_mode.pack(side="bottom",padx=10,pady=10)
        
        self.displayMode = 0 # slider 
        
        self.controller.bind('<KeyPress>',self.key_press)
        
    
    def changeMode(self):
        if self.displayMode == 0:
            self.displayMode = 1
            self.updateFromSlider(50)
        else:
            self.displayMode = 0
            self.updateFromSlider(50)
        
        
    def key_press(self,e):
        if e.keysym == "Right":
            value = self.slider.get()
            self.slider.set(value+SLIDER_SPEED)
            
        if e.keysym == "Left":
            value = self.slider.get()
            self.slider.set(value-SLIDER_SPEED)
            
        if e.keysym == "space":
            self.nextLink()
            
        if e.keysym == "m":
            self.changeMode()
            
        if e.keysym == "BackSpace":
            self.beforeLink()
        


                

        
    def setNewImg(self):
        id1,id2 = self.controller.orderedLinks[self.index]
        self.H = self.controller.Hm[id1,id2]
        
        folderPath = self.controller.folderPath
        nameList = self.controller.nameList
        
        path1 = Path(folderPath,nameList[id1])
        path2 = Path(folderPath,nameList[id2])
    
        self.img1 = cv.imread(str(path1)) # queryImage
        self.img2 = cv.imread(str(path2)) # trainImage
        
        self.labelCoin.config(text = f"liaison {self.index}/{self.numberLinks} \n {nameList[id1]}   -  {nameList[id2]} \n N = {int(self.controller.D[id1,id2])} ")
        self.slider.set(50)
        
    def updateFromSlider(self,e):
        if self.displayMode == 0:
            x = float(e)/100
            
            img = core.getSliderImg(self.img1,self.img2,self.H,x,ZOOM)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.labelImg.configure(image=img)
            self.labelImg.image=img
        
        else:
            h = ZOOM2
            a,b,c = np.shape(self.img1)
            img1 = cv.cvtColor(self.img1, cv.COLOR_BGR2RGB)[int(a/2-h/2):int(a/2+h/2),int(a/2-h/2):int(a/2+h/2),:]
            img2 = cv.cvtColor(self.img2, cv.COLOR_BGR2RGB)[int(a/2-h/2):int(a/2+h/2),int(a/2-h/2):int(a/2+h/2),:]

            img1 = Image.fromarray( img1)
            img2 = Image.fromarray(img2)
            dst = Image.new('RGB', (img1.width + img2.width, img1.height))
            dst.paste(img1, (0, 0))
            dst.paste(img2, (img1.width, 0))
            img = ImageTk.PhotoImage(dst)
            self.labelImg.configure(image=img)
            self.labelImg.image=img
            
            
        
    def nextLink(self):
        maxIdx = len(self.controller.orderedLinks)
        if self.index < maxIdx-1:
            self.index = self.index +1 
        
        self.setNewImg()
        self.updateFromSlider(50)
        
    def beforeLink(self):
        if self.index > 0:
            self.index = self.index -1 
        
        self.setNewImg()
        self.updateFromSlider(50)
        
class ComputationPage(tk.Frame):

    def __init__(self, parent, controller,folderPath):
        tk.Frame.__init__(self, parent)
        self.folderPath = folderPath
        self.controller = controller
        
        self.presentation = tk.StringVar()
        self.presentation.set("Appuyer sur start pour lancer le calcul :")
        
        label = tk.Label(self, textvariable=self.presentation)
        label.pack(side="top", fill="x", pady=5,padx=5)
        
        self.button1 = tk.Button(self, text="start",
                            command=threading.Thread(target=self.start,args=()).start)

        self.button1.pack(padx=5,pady=5)
        
        self.progress_var = tk.DoubleVar()
        
        progressBar = ttk.Progressbar(self,variable=self.progress_var, maximum=100)
        progressBar.pack(side="top",fill=tk.X,padx=10,pady=5)
        
        self.strAvance = tk.StringVar()
        labelAvanc = tk.Label(self, textvariable=self.strAvance)
        labelAvanc.pack(side="top", fill="x", pady=10)
        
        
        

        
        
    def start(self):
        import numpy as np
        
        self.presentation.set("Calcul en cours...")
        self.button1["state"]= tk.DISABLED
        
        self.timeCounter = 0 
        self.tmpsRestantMin = 0

        nameList, D ,Hm= core.getMatrixFromFolder(self.folderPath,
                                                  contrastThreshold=CONTRAST_THRESHOLD,
                                                  ratio=RATIO,callback=self.callbackProgressBar,
                                                  usePreprocessing=USE_PREPROCESSING,
                                                  preprocessingParam=preprocessingParam)

        np.save(Path(self.folderPath,"D.npy"),D)
        np.save(Path(self.folderPath,"Hm.npy"),Hm)
        np.save(Path(self.folderPath,"nameList.npy"),nameList)

        print("SAVING TO CSV")

        core.saveToCsv(self.folderPath,nameList,D)
                    
        nameList, D ,Hm = core.getSavedData(self.folderPath)
        orderedLinks = core.getOrderedLinks(D,Hm,USE_FILTER)
        
        self.controller.updateParam(nameList, D ,Hm,orderedLinks,self.folderPath)
        
        self.controller.frames["ComparePage"].setNewImg()
        self.controller.frames["ComparePage"].updateFromSlider(50)
        
        self.controller.show_frame("ComparePage")


                
    def callbackProgressBar(self,c,total):
        self.timeCounter += 1 
        pourcent = 100*c/total
        self.progress_var.set(pourcent)
        self.strAvance.set(f"{pourcent:.2f}% \n temps restant : {self.tmpsRestantMin:.2f}min")
        if self.timeCounter == 1 :
            self.t1 = time.time()
        if self.timeCounter == 50:
            delta = (time.time()-self.t1)/50 #par liaison
            
            tempsRestant = delta*(total-c)
            self.tmpsRestantMin = tempsRestant/60
            self.timeCounter = 0 
              
def start():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    app = App()
    app.mainloop()
    
    
    