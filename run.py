# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:23:19 2022

@author: vedhs
"""
import tkinter as tk
import iproc
import threading
from PIL import ImageTk, Image
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
        self._frame.grid(row=2,column=0,columnspan=2)
        
        
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
        
        
        #UI to show count image button
        self.UIe['countImages'] = tk.Button(self,text = "Count Images",command = self.__countImages)
        self.UIe['start'] = tk.Button(self,text = "Start",command = self.__start)
        self.UIe['countImages'].grid(row=3,column=0)
        self.UIe['start'].grid(row=3,column=1)
        
        
        #Canvas to show the image
        self.image=ImageTk.PhotoImage(Image.open('asset/image/preload.png').resize((512, 512),2))
        self.UIe['canvas']=tk.Label(self,image=self.image)
        self.UIe['canvas'].grid(row=4,column=0,columnspan=2)
        
        #To change the displayed image
        self.UIe['nextImage'] = tk.Button(self,text = "Save & Next",command = self.__countImages)
        self.UIe['prevImage'] = tk.Button(self,text = "Previous",command = self.__countImages)
        self.UIe['resetImage'] = tk.Button(self,text = "Reset",command = self.__start)
        self.UIe['nextImage'].grid(row=5,column=0,columnspan=2)
        self.UIe['prevImage'].grid(row=6,column=0)
        self.UIe['resetImage'].grid(row=6,column=1)
     
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
        path,batchN,augmentN = self.__getVars()
        self.__getImagePaths(path,batchN)
    
        
    #Get paths of the image which are going to be processed
    def __getImagePaths(self,path,batchN):
        self.imagePaths=[]
        def onStart():
            self.status.set('Getting image paths...')
        
        def getPaths(img,relpath,name):
            self.imagePaths.append({'path':relpath,'name':name})  
            
        def onEnd():
            self.status.set('Got the images...')
            
            #Set the first image
            self.image=ImageTk.PhotoImage(Image.open(path+'\\'+self.imagePaths[0]['path']+'\\'+self.imagePaths[0]['name']).resize((512, 512),2))
            self.UIe['canvas'].configure(image=self.image)
            
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
            if len(augmentNtxt)>0:
                augmentN=int(augmentNtxt)
            return path,batchN,augmentN
        except ValueError:
            self.status.set('BatchN and AugmentN must be an integer') 
            return '',0,0

if __name__=='__main__':
    gui=GUI()    
    gui.run()