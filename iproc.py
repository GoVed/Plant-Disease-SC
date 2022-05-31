# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:11:59 2022

@author: vedhs
"""

import os
import cv2
import time
import random
from pathlib import Path
import iconsole
import traceback

'''
Class to store data and process it
Data:
    info:bool=False     :   Print info while processing

Methods:
    proc:   Gets images from the folder with labels as immediated heirarchy folder name to the path      
        Args:
            func:function  : A function to be processed (pass it as object) 
            tkinterStatus=None   :   Tkinter status variable for GUI
        Return:
            None
            
    getImage:   Get a image from path, crop it to square and resize to given x and y
        Args:
            path:str    :   Path to the image to open
        Return:
            image:Numpy array   :   Processed image in numpy from the given path
            
    trainTestSplit:     Splits the given dataset with given ratio, saves in given path for train and test
        Args:
            trainPath:str   :   Path to save train set
            testPath:str    :   Path to save test set
            split:float=0.7   :   Split ratio, 0.7 means 70 in train and 30 in test
            tkinterStatus=None   :   Tkinter status variable for GUI
        Return:
            None
'''
class Data:
    
    def __init__(self,info:bool=False,path:str='',x:int=256,y:int=256,changeColor:bool=True):
        
        #Size of the image
        self.x=x
        self.y=y
        
        #Setting the data path to be processed on
        self.path=path
        
        #Print info bool
        self.info=info
        
        #Change from bgr to rgb for numpy processing
        self.changeColor=changeColor
    
    #Gets images from the folder with labels as immediated heirarchy folder name to the path  
    #func=None,tkinterStatus=None,randomBatchN:int=0,updateInfoEvery=10,loadImg=True,
    def proc(self,data):
        
        #Initialize loader
        if self.info: 
            loader = iconsole.Loader("Processing images", "Processed images")
                        
        try:
                      
            #Get all directories in the path, take them as labels
            labels=[name for name in os.listdir(self.path)]
            
            #To count the speed of processing
            lt=time.time()
            updateCheck=0
            updateInfoEvery=10
            
            if 'onStart' in data:
                data['onStart']()
            
            if 'updateInfoEvery' in data:
                updateInfoEvery=data['updateInfoEvery']
            for label in labels:
                
                #For each label get all files from the subdirectories
                for ipath, subdirs, files in os.walk(os.path.join(self.path, label)):
                    if 'randomBatchN' in data:
                        if len(files)>data['randomBatchN']:
                            random.shuffle(files)
                            files=files[:data['randomBatchN']]                    
                    for name in files:                          
                        if self.info:
                            if updateCheck>=updateInfoEvery:
                                status=os.path.join(ipath, name)+' '+(str(round(updateInfoEvery/(time.time()-lt)))+'/s') if time.time()>lt else ''
                                loader.desc=status
                                if 'tkinterStatus' in data:
                                    data['tkinterStatus'].set(status+'\nTime elapsed: '+loader.elapsedTime)                                                                
                                lt=time.time()
                                updateCheck=0
                            else:
                                updateCheck+=1
                        
                        #Get image from the path
                        img=None
                        if ('loadImg' in data) and (data['loadImg']):
                            img=self.getImage(os.path.join(ipath, name))
                        
                        #Calling the process function
                        if 'func' in data:
                            data['func'](img,os.path.relpath(ipath, self.path),name)
                        
            if 'tkinterStatus' in data:
                data['tkinterStatus'].set('Proccessed Images') 
                
            if 'onEnd' in data:
                data['onEnd']()
        
        except Exception :
            print('\nFailed to get Images\n',traceback.print_exc())
                        
        #Stop the loader
        if self.info:             
            loader.stop()
       
    def getImage(self,path):
        #Reading the image then changing to rgb and resizing it
        img=cv2.imread(path)
                    
        #Cropping to square
        w,h=img.shape[0],img.shape[1]
        if w!=h:
            if w<h:
                img=img[:,int(h/2)-int(w/2):int(h/2)+int(w/2),:]
            else:
                img=img[int(w/2)-int(h/2):int(w/2)+int(h/2),:,:]
        
        #Resizing to given x and y
        img=cv2.resize(img,(self.x,self.y))
        
        if self.changeColor:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
        return img
            
    # Splits the given dataset with given ratio
    def trainTestSplit(self,trainPath:str,testPath:str,split:float=0.7,tkinterStatus=None):
        
        #Making a temporary function to be passed for processing        
        def _temp(img,relpath,name):
            
            #Check if path exists or creats it
            Path(os.path.join(trainPath,relpath)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(testPath,relpath)).mkdir(parents=True, exist_ok=True)
            
            #Check where to save, train or test set
            saveAt=random.random()         

            #Copy the name from parent set               
            name=os.path.join(relpath,name)
            
            #Save operation
            cv2.imwrite(os.path.join(trainPath,name), img) if saveAt<split else cv2.imwrite(os.path.join(testPath,name), img)
            
        #Call to process func            
        self.proc({'func':_temp,'tkinterStatus':tkinterStatus,'loadImg':True})
        

  
if __name__=='__main__':
    td=Data(info=True,path='data\\raw')
    count=0
    def temp(img,relpath,name):
        global count
        count+=1
    td.proc(temp,randomBatchN=5,loadImg=False)
    print(count)
    
    
   
    