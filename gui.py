import tkinter as tk   
import tkinter.filedialog as fd  
import tkinter.ttk as ttk  

from pathlib import Path  

import cv2 as cv
from PIL import Image,ImageTk

import threading
import configparser
import sys

### Lecture des paramètres depuis config.ini
config = configparser.ConfigParser()
parentPath = Path(Path(__file__).parent)
sys.path.append(parentPath)
import core

config.read(Path(parentPath,"config.ini"))

SLIDER_SPEED = int(config["controls"]["sliderSpeed"])
CONTRAST_THRESHOLD = float(config["calcul"]["contrastThreshold"])
RATIO = float(config["calcul"]["ratio"])
ZOOM = int(config["graphics"]["zoom"])


class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)


        ### top frame
        topFrame = tk.Frame(self,highlightbackground="black", highlightthickness=1)
        topFrame.pack(side="top",fill=tk.X)
        
        titleLabel = tk.Label(master=topFrame,text="Outil de classification",padx=5,pady=5)
        titleLabel.pack(side="left")
        
        versionLabel = tk.Label(master=topFrame,text="V1 02/02/2023")
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
            
            print("data available")
        
            nameList, D ,Hm = core.getSavedData(folderPath)
            orderedLinks = core.getOrderedLinks(D,Hm)
            
            self.controller.updateParam(nameList, D ,Hm,orderedLinks,folderPath)
            
            self.controller.frames["ComparePage"].setNewImg()
            self.controller.frames["ComparePage"].updateFromSlider(50)
            
            self.controller.show_frame("ComparePage")
            
        else:
            self.controller.startNewComputation(folderPath)
        
class ComparePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        self.controller = controller
        
        self.index = 0 
        
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
        
        self.controller.bind('<KeyPress>',self.key_press)
        
        
    def key_press(self,e):
        if e.keysym == "Right":
            value = self.slider.get()
            self.slider.set(value+SLIDER_SPEED)
            
        if e.keysym == "Left":
            value = self.slider.get()
            self.slider.set(value-SLIDER_SPEED)
            
        if e.keysym == "space":
            self.nextLink()
            
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
        
        self.labelCoin.config(text = f"{nameList[id1]}   -  {nameList[id2]} \n D = {int(self.controller.D[id1,id2])} ")
        self.slider.set(50)
        
    def updateFromSlider(self,e):
        x = float(e)/100
        
        img = core.getSliderImg(self.img1,self.img2,self.H,x,ZOOM)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
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
        label = tk.Label(self, text="Appuyer sur start pour lancer le calcul")
        label.pack(side="top", fill="x", pady=5,padx=5)
        
        self.progress_var = tk.DoubleVar()
        
        progressBar = ttk.Progressbar(self,variable=self.progress_var, maximum=100)
        progressBar.pack(side="top",fill=tk.X,padx=5,pady=5)
        
        self.strAvance = tk.StringVar()
        
        labelAvanc = tk.Label(self, textvariable=self.strAvance)
        labelAvanc.pack(side="top", fill="x", pady=10)
        
        
        
        button1 = tk.Button(self, text="start",
                            command=threading.Thread(target=self.start,args=()).start)

        button1.pack(padx=20,pady=20)
        
        
    def start(self):
        import numpy as np

        nameList, D ,Hm= core.getMatrixFromFolder(self.folderPath,contrastThreshold=CONTRAST_THRESHOLD,ratio=RATIO,callback=self.callbackProgressBar)

        np.save(Path(self.folderPath,"D.npy"),D)
        np.save(Path(self.folderPath,"Hm.npy"),Hm)
        np.save(Path(self.folderPath,"nameList.npy"),nameList)
        
        
        self.controller.show_frame("StartPage")
        

                
    def callbackProgressBar(self,val):
        self.progress_var.set(val)
        self.strAvance.set(f"{val:.2f}%")
        
        
def start():
    app = App()
    app.mainloop()
    

if __name__ == "__main__":
    app = App()
    app.mainloop()
    
    
    