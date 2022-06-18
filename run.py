# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:23:19 2022

@author: vedhs
"""
import tkinter as tk
import iproc
import threading
from PIL import ImageTk, Image
import numpy as np
'''
Class to make Graphical User Interface

Credits:
    GUI frame logic     :   Stevoisiak [https://stackoverflow.com/users/3357935/stevoisiak]
    
GUI:    Makes parent GUI and opens train test split tab
    None
run:    Runs the mainloop  
    Args:
        None
    Return:
        None
        
'''
class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame=None

        #Init for UI elements to show on tk window
        self.UIe = {}
        
        #Label to show status
        self.status = tk.StringVar()
        self.UIe['status'] = tk.Label(self,textvariable= self.status)
        self.UIe['status'].grid(row=0,column=0,columnspan=2)
        self.status.set("Choose what to do")
        
        #Buttons to switch frames
        self.UIe['setTrainTestFrame']=tk.Button(self,text='Split Dataset',command=lambda:self.switch_frame(TrainTestFrame)).grid(row=1,column=0)
        self.UIe['setTrainTestFrame']=tk.Button(self,text='Manually Segment',command=lambda:self.switch_frame(ManualSegmentFrame)).grid(row=1,column=1)
        self.UIe['setModelTraining']=tk.Button(self,text='Train Model',command=lambda:self.switch_frame(TrainModelFrame)).grid(row=1,column=2)
        #Set initial frame to Train Test Frame
        self.switch_frame(TrainTestFrame)
        
    #Run the mainloop
    def run(self):
        self.mainloop()
        
    #Destroy the previous frame and adds new frame to the UI
    def switch_frame(self,frame):
        if self._frame is not None:
            self._frame.destroy()
        self._frame=frame(self)
        self._frame.grid(row=2,column=0,columnspan=3)
        
        
'''
TrainTestFrame:     Makes the frame for splitting dataset into train and test dataset
    None
'''      
class TrainTestFrame(tk.Frame):
    #Constructor
    def __init__(self,master):        
        tk.Frame.__init__(self, master) 
        self.status = master.status
        #Init for UI elements to show on tk window
        self.UIe = {}
        
        self.px,self.py=0,0       
                        
        #Getting the data path from the user
        self.UIe['pathLabel'] = tk.Label(self, text = 'Path:')   
        self.UIe['trainPathLabel'] = tk.Label(self, text = 'Train path:')   
        self.UIe['testPathLabel'] = tk.Label(self, text = 'Test path:')
        self.UIe['splitRatioLabel'] = tk.Label(self, text = 'Split ratio:')        
          
        #Input fields for user to enter values
        self.UIe['path'] = tk.Text(self,height = 1,width = 20)        
        self.UIe['trainPath'] = tk.Text(self,height = 1,width = 20)        
        self.UIe['testPath'] = tk.Text(self,height = 1,width = 20)
        self.UIe['splitRatio'] = tk.Text(self,height = 1,width = 20)
        
        #Set labels on the UI
        self.UIe['pathLabel'].grid(row=0,column=0)
        self.UIe['path'].grid(row=0,column=1)
        self.UIe['trainPathLabel'].grid(row=1,column=0)
        self.UIe['trainPath'].grid(row=1,column=1)
        self.UIe['testPathLabel'].grid(row=2,column=0)
        self.UIe['testPath'].grid(row=2,column=1)                
        self.UIe['splitRatioLabel'].grid(row=3,column=0)
        self.UIe['splitRatio'].grid(row=3,column=1)
        
        #UI button to train test split button
        self.UIe['ttsplit'] = tk.Button(self,text = "Train Test Split",command = self.__runTrainTestSplit)
        self.UIe['ttsplit'].grid(row=4,column=0,columnspan=2)
                
        
    def __runTrainTestSplit(self):
        #Get the text from the text fields
        path=self.UIe['path'].get(1.0, "end-1c")
        trainPath=self.UIe['trainPath'].get(1.0, "end-1c")
        testPath=self.UIe['testPath'].get(1.0, "end-1c")
        splitRatio=self.UIe['splitRatio'].get(1.0, "end-1c")
        
        try:
            #validating the data
            splitRatio=float(splitRatio)
            if len(path)>0 and len(trainPath)>0 and len(testPath)>0:
                #Calling proc function with the given data and running it on different thread
                data=iproc.Data(info=True,path=path,changeColor=False)
                
                trainTestThread=threading.Thread(target=data.trainTestSplit,args=(trainPath, testPath, splitRatio,self.status,))
                trainTestThread.setDaemon(True)
                trainTestThread.start()                
            else:
                self.status.set('Fill path,train path and test path')
        except ValueError:
            self.status.set('Split ratio must be a number')      

'''      
ManualSegmentFrame:     To manually segment required images for training segmentation model
    None
'''      
class ManualSegmentFrame(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.status = master.status
        
        self.UIe={}
        
        #Getting the data path from the user and setting it onto the UI
        self.UIe['pathLabel'] = tk.Label(self, text = 'Path:')  
        self.UIe['path'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['pathLabel'].grid(row=0,column=0)
        self.UIe['path'].grid(row=0,column=1)
        
        
        #UI to show batch label and input
        self.UIe['batchNLabel'] = tk.Label(self, text = 'Images per class:')  
        self.UIe['batchN'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['batchNLabel'].grid(row=1,column=0)
        self.UIe['batchN'].grid(row=1,column=1)
        
        
        #UI to show augment label and input
        self.UIe['augmentNLabel'] = tk.Label(self, text = 'Augment N:')  
        self.UIe['augmentN'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['augmentNLabel'].grid(row=2,column=0)
        self.UIe['augmentN'].grid(row=2,column=1)
        
        #UI to show save path label and input
        self.UIe['savePathLabel'] = tk.Label(self, text = 'Save Path:')  
        self.UIe['savePath'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['savePathLabel'].grid(row=3,column=0)
        self.UIe['savePath'].grid(row=3,column=1)
        
        #UI to show count image button
        self.UIe['countImages'] = tk.Button(self,text = "Count Images",command = self.__countImages)
        self.UIe['start'] = tk.Button(self,text = "Start",command = self.__start)
        self.UIe['countImages'].grid(row=4,column=0)
        self.UIe['start'].grid(row=4,column=1)
                
        #Canvas to show the image
        self.image=ImageTk.PhotoImage(Image.open('asset/image/preload.png').resize((512, 512),2))
        self.UIe['canvas']=tk.Label(self,image=self.image)
        self.UIe['canvas'].grid(row=5,column=0,columnspan=2)
        
        #To change the displayed image
        self.UIe['nextImage'] = tk.Button(self,text = "Save & Next",command = self.__loadNextImage)
        self.UIe['prevImage'] = tk.Button(self,text = "Previous",command = self.__loadPrevImage)
        self.UIe['resetImage'] = tk.Button(self,text = "Reset",command = self.__start)
        self.UIe['nextImage'].grid(row=6,column=0,columnspan=2)
        self.UIe['prevImage'].grid(row=7,column=0)
        self.UIe['resetImage'].grid(row=7,column=1)
     
    #To count how many images are going to be shown with given parameters
    def __countImages(self):
        
        #Get inputs
        path,batchN,augmentN = self.__getVars()
        
        
        def onStart():
            self.status.set('Loading...')
        
        def temp(img,relpath,name):                 
            self.count+=1
            
        def onEnd():
            self.status.set(f'Total images to manually segment:{self.count}\nOutput images:{self.count*(augmentN+1)}')
            
        #Making object and setting count to 0
        td=iproc.Data(path=path)
        self.count=0
        
        #Start the processing thread
        countThread=threading.Thread(target=td.proc,args=({'func':temp,'randomBatchN':batchN,'onEnd':onEnd,'onStart':onStart},))
        countThread.setDaemon(True)
        countThread.start()
      
        
    
    def __start(self):
        #Get input variables
        self.path,self.batchN,self.augmentN = self.__getVars()
        
        #Get images paths to be processed
        self.__getImagePaths(self.path,self.batchN)
        
    #On mouse move
    def _motion(self,event):
        #Get mouse coordinates and bound them to 0-255
        self.py, self.px = round((event.x-2)/2), round((event.y-2)/2)
        if self.px>255:
            self.px=255
        if self.px<0:
            self.px=0
        if self.py>255:
            self.py=255
        if self.py<0:
            self.py=0
        
        #Update status
        self.status.set(str(self._currentImage+1)+'/'+str(len(self.imagePaths))+'\tMouse X:'+str(self.px)+'\tY:'+str(self.py)+'\tCursor size '+str(self._cursorSize))
        
        #Calculate the modfying rea using the cursor size
        fx=max(0,self.px-self._cursorSize)
        tx=min(self.mask.shape[0],self.px+self._cursorSize)
        fy=max(0,self.py-self._cursorSize)
        ty=min(self.mask.shape[1],self.py+self._cursorSize)
        
        #left click
        if event.state%33==0:
            self.mask[fx:tx,fy:ty]=255
            self.__loadCurrImage()
            
        #right click
        if event.state%129==0:
            self.mask[fx:tx,fy:ty]=0
            self.__loadCurrImage()
     
    #on mouse scroll
    def _onScroll(self,event):
        #scroll up and down to change cursor size
        if event.delta>0:
            self._cursorSize+=1
        else:
            if self._cursorSize>0:
                self._cursorSize-=1  
                
        self.status.set(str(self._currentImage+1)+'/'+str(len(self.imagePaths))+'\tMouse X:'+str(self.px)+'\tY:'+str(self.py)+'\tCursor size '+str(self._cursorSize))
         
    #Calculate HSV mask using the current image
    def _setHSVMask(self):
        self.mask=iproc.threshMask(iproc.rgb_to_hsv(self.npimage),lowerbound=np.array([0.02,0,0]),upperbound=np.array([0.5,0,0]),mode=0).reshape(256,256,1)
      
    #Get the image in numpy array
    def _setNpImage(self):
        self.npimage=iproc.getImage(self.imagePaths[self._currentImage]['root']+'\\'+self.imagePaths[self._currentImage]['path']+'\\'+self.imagePaths[self._currentImage]['name'],changeColor=True)
        
    #From numpy image and hsv mask, calculate the image to show
    def __loadCurrImage(self):      
        _=(self.npimage/2)+(self.npimage/4*self.mask)+((np.full(self.npimage.shape,1)*np.array([255/4,0,0]))*self.mask)
        self.image=ImageTk.PhotoImage(Image.fromarray((_).astype(np.uint8)).resize((512, 512),2))
        self.UIe['canvas'].configure(image=self.image)
    
    #Saves the image
    def __saveImage(self):
        
        ai,am=iproc.augment(self.npimage,self.mask[:,:,0],path='data/raw/Negative/Negative/',rotate=True,zoom=True,bg=True,n=self.augmentN)
        for i in range(len(ai)):            
            iproc.saveImage(ai[i], self._savePath+'\\image\\'+self.imagePaths[self._currentImage]['path'],str(i)+self.imagePaths[self._currentImage]['name'])
            iproc.saveImage(am[i], self._savePath+'\\mask\\'+self.imagePaths[self._currentImage]['path'],str(i)+self.imagePaths[self._currentImage]['name'],changeToBGR=False)
        self.status.set('Saved')
        
    #loads next image in the image list
    def __loadNextImage(self):
        self.__saveImage()
        if self._currentImage<len(self.imagePaths)-1:
            self._currentImage+=1
            self._setNpImage()
            self._setHSVMask()
            self.__loadCurrImage()
            
            
        else:
            self.status.set('Last image reach, start with a new batch of image')
        
    #loads prev image in the image list            
    def __loadPrevImage(self):
        if self._currentImage>0:
            self._currentImage-=1
            self._setNpImage()
            self._setHSVMask()
            self.__loadCurrImage()
            
        else:
            self.status.set('First image reached!')
            
            
    #Get paths of the image which are going to be processed
    def __getImagePaths(self,path,batchN):
        self.imagePaths=[]
        def onStart():
            self.status.set('Getting image paths...')
        
        def getPaths(img,relpath,name):
            self.imagePaths.append({'path':relpath,'name':name,'root':path})  
            
        def onEnd():
            self.status.set('Got the images...')
            self._currentImage=0
            self._cursorSize=2
            self._setNpImage()
            self._setHSVMask()
            #Set the first image
            self.__loadCurrImage()
            
            self.UIe['canvas'].bind('<Motion>', self._motion)
            self.UIe['canvas'].bind("<MouseWheel>", self._onScroll)
            
        #making the processing object
        td=iproc.Data(path=path)
        
        #Start the processing thread
        getPathThread=threading.Thread(target=td.proc,args=({'func':getPaths,'randomBatchN':batchN,'onEnd':onEnd,'onStart':onStart},))
        getPathThread.setDaemon(True)
        getPathThread.start()        
      
    #To get the inputted values by the user
    def __getVars(self):
        path=self.UIe['path'].get(1.0, "end-1c")
        try:
            batchN=int(self.UIe['batchN'].get(1.0, "end-1c"))
            augmentN=0
            augmentNtxt=self.UIe['augmentN'].get(1.0, "end-1c")
            self._savePath=self.UIe['savePath'].get(1.0, "end-1c")
            if len(self._savePath)>0:
                if len(augmentNtxt)>0:
                    augmentN=int(augmentNtxt)
                return path,batchN,augmentN
            else:
                self.status.set('Enter the save path')
                return '',0,0
        except ValueError:
            self.status.set('BatchN and AugmentN must be an integer') 
            return '',0,0
class TrainModelFrame(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        
if __name__=='__main__':
    gui=GUI()    
    gui.run()