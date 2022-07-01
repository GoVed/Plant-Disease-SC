# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:23:19 2022

@author: vedhs
"""
import tkinter as tk
import iproc
import threading
import ml
from PIL import ImageTk, Image
import numpy as np
'''
Credits:
    GUI frame logic     :   Stevoisiak [https://stackoverflow.com/users/3357935/stevoisiak]
'''

    
'''
Class to make Graphical User Interface

    
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
        self.UIe['status'].grid(row=0,column=0,columnspan=4)
        self.status.set("Choose what to do")
        
        #Buttons to switch frames
        self.UIe['setTrainTestFrame']=tk.Button(self,text='Split Dataset',command=lambda:self.switch_frame(TrainTestFrame)).grid(row=1,column=0)
        self.UIe['setManuallySegment']=tk.Button(self,text='Manually Segment',command=lambda:self.switch_frame(ManualSegmentFrame)).grid(row=1,column=1)
        self.UIe['setLoadSegmented']=tk.Button(self,text='Load Segmented',command=lambda:self.switch_frame(LoadSegmentFrame)).grid(row=1,column=2)
        self.UIe['setModelTraining']=tk.Button(self,text='Train Model',command=lambda:self.switch_frame(TrainModelFrame)).grid(row=1,column=3)
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
TrainTestFrame:     
    Makes the frame for splitting dataset into train and test dataset
    Args:
        master:
            Main tkinter window to hold the frame
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
ManualSegmentFrame:     
    To manually segment required images for training segmentation model
    Args:
        master:
            Main tkinter window to hold the frame
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
        
class LoadSegmentFrame(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.status = master.status
        
        self.UIe={}
        
        #Getting the data path from the user and setting it onto the UI
        self.UIe['pathLabel'] = tk.Label(self, text = 'Path:')  
        self.UIe['path'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['pathLabel'].grid(row=0,column=0)
        self.UIe['path'].grid(row=0,column=1)
        
        self.UIe['path'].insert(tk.INSERT,'data/color')
        
        
        #UI to show batch label and input
        self.UIe['segmentedPathLabel'] = tk.Label(self, text = 'Segmented Path:')  
        self.UIe['segmentedPath'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['segmentedPathLabel'].grid(row=1,column=0)
        self.UIe['segmentedPath'].grid(row=1,column=1)
        
        self.UIe['segmentedPath'].insert(tk.INSERT,'data/segmented')
        

        self.UIe['segmentedPathSuffixLabel'] = tk.Label(self, text = 'Segmented Path Suffix:')  
        self.UIe['segmentedPathSuffix'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['segmentedPathSuffixLabel'].grid(row=2,column=0)
        self.UIe['segmentedPathSuffix'].grid(row=2,column=1)
        
        self.UIe['segmentedPathSuffix'].insert(tk.INSERT,'_final_masked')
        
        #UI to show save path label and input
        self.UIe['savePathTrainLabel'] = tk.Label(self, text = 'Save Path (Train):')  
        self.UIe['savePathTrain'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['savePathTrainLabel'].grid(row=3,column=0)
        self.UIe['savePathTrain'].grid(row=3,column=1)
        
        self.UIe['savePathTrain'].insert(tk.INSERT,'data/train/manseg')
        
        self.UIe['savePathTestLabel'] = tk.Label(self, text = 'Save Path (Test):')  
        self.UIe['savePathTest'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['savePathTestLabel'].grid(row=4,column=0)
        self.UIe['savePathTest'].grid(row=4,column=1)
        
        self.UIe['savePathTest'].insert(tk.INSERT,'data/test/manseg')
        
        
        #UI to show split ratio
        self.UIe['splitRatioLabel'] = tk.Label(self, text = 'Split Ratio:')  
        self.UIe['splitRatio'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['splitRatioLabel'].grid(row=5,column=0)
        self.UIe['splitRatio'].grid(row=5,column=1)
        
        self.UIe['splitRatio'].insert(tk.INSERT,'0.7')
        
        #UI to show buttons
        self.UIe['process'] = tk.Button(self,text = "Process",command = self.__process)
        
        self.UIe['process'].grid(row=6,column=0)
        
        
    #To get the inputted values by the user
    def __getVars(self):
        path=self.UIe['path'].get(1.0, "end-1c")
        segmentedPath=self.UIe['segmentedPath'].get(1.0, "end-1c")
        segmentedPathSuffix=self.UIe['segmentedPathSuffix'].get(1.0, "end-1c")
        savePathTrain=self.UIe['savePathTrain'].get(1.0, "end-1c")
        savePathTest=self.UIe['savePathTest'].get(1.0, "end-1c")
        try:                     
            splitRatio=float(self.UIe['splitRatio'].get(1.0, "end-1c"))
            return path,segmentedPath,segmentedPathSuffix,savePathTrain,savePathTest,splitRatio
        except:
            print('Split Ratio needs to be a number')
            return'','','','','',0
        
    
    def __process(self):
        path,segmentedPath,segmentedPathSuffix,savePathTrain,savePathTest,splitRatio=self.__getVars()
        data=iproc.Data(path=path,changeColor=False)
        
        def imgproc(img,relpath,name):
            try:
                fpath=segmentedPath+'/'+relpath+'/'+name[:name.rindex('.')]+segmentedPathSuffix+name[name.rindex('.'):]            
                mask=iproc.getImage(fpath,(data.x,data.y),data.changeColor)
                mask=np.sum(mask,axis=2)
                mask[mask>5]=255
                mask=iproc.removeNoiseMask(mask)
                if not iproc.checkIfFileExist([savePathTrain+'/image/'+relpath+'/'+name,savePathTest+'/image/'+relpath+'/'+name]):
                    print('Saving',savePathTrain+'/image/'+relpath+'/'+name)
                    if iproc.binaryProb(splitRatio):
                        iproc.saveImage(img, savePathTrain+'/image/'+relpath,name,changeToBGR=False)
                        iproc.saveImage(mask, savePathTrain+'/mask/'+relpath,name,changeToBGR=False,isGray=True)
                    else:
                        iproc.saveImage(img, savePathTest+'/image/'+relpath,name,changeToBGR=False)
                        iproc.saveImage(mask, savePathTest+'/mask/'+relpath,name,changeToBGR=False,isGray=True)
                else:
                    print('Skipping',savePathTrain+'/image/'+relpath+'/'+name)
            except:
                print(savePathTrain+'/image/'+relpath+'/'+name,'not saved')
                
        #Setting the thread    
        thread=threading.Thread(target=data.proc,args=({'func':imgproc,'loadImg':True},))
        thread.setDaemon(True)
        thread.start() 
        
        
'''
TrainModelFrame:
    Frame to hold the train model tab
    Args:
        master:
            Main tkinter window to hold the frame
'''
class TrainModelFrame(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.status = master.status
        
        self.UIe={}
        
        #Getting the data path from the user and setting it onto the UI
        self.UIe['trainFeaturePathLabel'] = tk.Label(self, text = 'Train Features Path:')  
        self.UIe['trainFeaturePath'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['trainFeaturePathLabel'].grid(row=0,column=0)
        self.UIe['trainFeaturePath'].grid(row=0,column=1)
        
        self.UIe['trainFeaturePath'].insert(tk.INSERT,'data/train/manseg/image')
        
        self.UIe['trainLabelPathLabel'] = tk.Label(self, text = 'Train Labels Path:')  
        self.UIe['trainLabelPath'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['trainLabelPathLabel'].grid(row=1,column=0)
        self.UIe['trainLabelPath'].grid(row=1,column=1)
        
        self.UIe['trainLabelPath'].insert(tk.INSERT,'data/train/manseg/mask')
        
        self.UIe['testFeaturePathLabel'] = tk.Label(self, text = 'Test Features Path:')  
        self.UIe['testFeaturePath'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['testFeaturePathLabel'].grid(row=2,column=0)
        self.UIe['testFeaturePath'].grid(row=2,column=1)
        
        self.UIe['testFeaturePath'].insert(tk.INSERT,'data/test/manseg/image')

        self.UIe['testLabelPathLabel'] = tk.Label(self, text = 'Test Labels Path:')  
        self.UIe['testLabelPath'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['testLabelPathLabel'].grid(row=3,column=0)
        self.UIe['testLabelPath'].grid(row=3,column=1)
        
        self.UIe['testLabelPath'].insert(tk.INSERT,'data/test/manseg/mask')
        
        self.UIe['modelSavePathLabel'] = tk.Label(self, text = 'Model Save Path:')  
        self.UIe['modelSavePath'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['modelSavePathLabel'].grid(row=4,column=0)
        self.UIe['modelSavePath'].grid(row=4,column=1)
        
        self.UIe['modelSavePath'].insert(tk.INSERT,'model/segment')

        self.UIe['epochsLabel'] = tk.Label(self, text = 'Epochs:')  
        self.UIe['epochs'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['epochsLabel'].grid(row=5,column=0)
        self.UIe['epochs'].grid(row=5,column=1)
        
        self.UIe['batchSizeLabel'] = tk.Label(self, text = 'Batches:')  
        self.UIe['batchSize'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['batchSizeLabel'].grid(row=6,column=0)
        self.UIe['batchSize'].grid(row=6,column=1)
        
        self.UIe['imageNLabel'] = tk.Label(self, text = 'Image per folder:')  
        self.UIe['imageN'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['imageNLabel'].grid(row=7,column=0)
        self.UIe['imageN'].grid(row=7,column=1)
        
        self.UIe['startFromLabel'] = tk.Label(self, text = 'Start from:')  
        self.UIe['startFrom'] = tk.Text(self,height = 1,width = 20) 
        self.UIe['startFromLabel'].grid(row=8,column=0)
        self.UIe['startFrom'].grid(row=8,column=1)
        #Drop down list to select the type of the model
        self.modelType=tk.StringVar()
        self.modelType.set("Select model")
        self.UIe['modelTypeLabel'] = tk.Label(self, text = 'Model Type:')  
        self.UIe['modelType'] = tk.OptionMenu(self, self.modelType,'Segmentation Model')
        self.UIe['modelTypeLabel'].grid(row=9,column=0)
        self.UIe['modelType'].grid(row=9,column=1)
        
        #UI to show buttons
        self.UIe['trainFull'] = tk.Button(self,text = "Train All",command = self.__trainAll)
        self.UIe['trainFolder'] = tk.Button(self,text = "Train Folder",command = self.__trainFolder)
        self.UIe['trainFull'].grid(row=10,column=0)
        self.UIe['trainFolder'].grid(row=10,column=1)
        
    '''
    __trainAll:
        to train all the images folder wise
        Args:
            None
        Return:
            None
    '''
    def __trainAll(self):
        trainFeaturePath,trainLabelPath,testFeaturePath,testLabelPath,modelSavePath,epochs,batchN,imageN,startFrom=self.__getVars()
        acs=[]
        vacs=[]
        folders=None
        if trainFeaturePath!='':
            folders=iproc.getFolders(trainFeaturePath)
            for folder in folders:
                if startFrom<=0:
                    print('Training',folder)
                    ac,vac=self.__trainFolder(folder)
                    acs.append(ac)
                    vacs.append(vac)
                else:
                    print('Skipping',folder)
                    startFrom-=1
        with open(modelSavePath+'/acc.csv', 'w') as f:
            f.writelines('Folder,Model,Batch,Epoch,Acc,Val Acc;')
            
            for i in range(len(folders)):
                for j in range(len(acs[i])):
                    for k in range(len(acs[i][j])):
                        for l in range(len(acs[i][j][k])):
                            f.writelines(folders[i]+','+str(j)+','+str(k)+','+str(l)+','+str(acs[i][j][k][l])+','+str(vacs[i][j][k][l])+'\n')
                        
                        
                 
    '''
    __trainFolder:
        to train a single folder
        Args:
            takePath:str
                the path for model to be saved in
        Return:
            None
    '''
    def __trainFolder(self,takePath:str=''):
        
        trainFeaturePath,trainLabelPath,testFeaturePath,testLabelPath,modelSavePath,epochs,batchN,imageN,startFrom=self.__getVars()
        if takePath!='':
            trainFeaturePath+='/'+takePath
            trainLabelPath+='/'+takePath
            testFeaturePath+='/'+takePath
            testLabelPath+='/'+takePath
        if trainFeaturePath!='':
            modelSavePathFinal=modelSavePath
            if takePath!='':
                modelSavePathFinal=modelSavePath+'/'+takePath
            ac,vac=self.__train(trainFeaturePath,trainLabelPath,testFeaturePath,testLabelPath,batchN,imageN,modelSavePathFinal,epochs)
        return ac,vac
            
                   
    '''
    __getVars:
        get the value from the text field in the UI
        Args:
            None
        Return:
            trainFeaturePath:str
                path for training features
            trainLabelPath:str
                path for training labels
            testFeaturePath:str
                path for testing features
            testLabelPath:str
                path for testing labels
            modelSavePath:str
                path for sving the model
            epochs:int
                number of epochs to be trained for
            batchN:int
                number of batches for training
            imageN:int
                number of images per batch per folder
            startFrom:int
                for training multiple folders and start from a point skipping n folders for training
    '''
    def __getVars(self):
        trainFeaturePath=self.UIe['trainFeaturePath'].get(1.0, "end-1c")
        trainLabelPath=self.UIe['trainLabelPath'].get(1.0, "end-1c")
        testFeaturePath=self.UIe['testFeaturePath'].get(1.0, "end-1c")
        testLabelPath=self.UIe['testLabelPath'].get(1.0, "end-1c")
        modelSavePath=self.UIe['modelSavePath'].get(1.0, "end-1c")
        epochs=self.UIe['epochs'].get(1.0, "end-1c")
        batchN=self.UIe['batchSize'].get(1.0, "end-1c")
        imageN=self.UIe['imageN'].get(1.0, "end-1c")
        startFrom=self.UIe['startFrom'].get(1.0, "end-1c")
        if len(trainFeaturePath)>0 and len(trainLabelPath)>0 and len(testFeaturePath)>0 and len(testLabelPath)>0 and len(modelSavePath)>0:
            try:
                if len(epochs)<=0:
                    epochs=10
                    
                if len(batchN)<=0:
                    batchN=25    
                                        
                if len(imageN)<=0:
                    imageN=5
                    
                if len(startFrom)<=0:
                    startFrom=0
                    
                epochs=int(epochs)
                batchN=int(batchN)
                imageN=int(imageN)
                startFrom=int(startFrom)
                return trainFeaturePath,trainLabelPath,testFeaturePath,testLabelPath,modelSavePath,epochs,batchN,imageN,startFrom
            except:
                self.status.set('Epochs, Batch size, Image per Folder and Start From must be an integer') 
                return '','','','','',0,0,0,0
        else:            
            self.status.set('Enter all the fields')
            return '','','','','',0,0,0,0
    
    '''
    __train:
        trains the model with given parameters
        Args:
            trainFeaturePath:str
                path for training features
            trainLabelPath:str
                path for training labels
            testFeaturePath:str
                path for testing features
            testLabelPath:str
                path for testing labels
            batchN:int
                total random batches to be trained
            imageN:int
                images per folder to be taken  
            modelSavePath:str=''
                the path for medl to be saved at
        Return:
            None
    '''
    def __train(self,trainFeaturePath:str,trainLabelPath:str,testFeaturePath:str,testLabelPath:str,batchN:int,imageN:int,modelSavePath='',epochs:int=10):
        self.mlTrain=ml.SegmentML()        
        self.mlTrain.compileModel()
        if modelSavePath!='':
            self.mlTrain.modelSavePath=modelSavePath
        ac,vac=self.mlTrain.trainBatchWise(trainFeaturePath,trainLabelPath,testFeaturePath,testLabelPath,batchN,imageN,epochs)
        return ac,vac
        
        
if __name__=='__main__':
    gui=GUI()    
    gui.run()