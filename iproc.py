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
from matplotlib.colors import rgb_to_hsv
import numpy as np
import numba as nb
from scipy import ndimage
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
                            img=getImage(os.path.join(ipath, name),(self.x,self.y),self.changeColor)
                        
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


'''
getImage:   Get a image from path, crop it to square and resize to given x and y
    Args:
        path:str    :   Path to the image to open
    Return:
        image:Numpy array   :   Processed image in numpy from the given path
'''        
def getImage(path,size=(0,0),changeColor:bool=False):
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
    if size!=(0,0):
        img=cv2.resize(img,size)
    
    if changeColor:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
    return img
 
'''
threshMask (uses numba njit)
    It generates a mask using the threshold value with given 3 channel image
Args:
    lowerbound:np array:    Lowerbound for threshold
    upperbound:np array:    Upperbound for threshold
    mode:int:
        -2      Check all threshold limits (and)
        -1      Check any threshold limit (or)
        0/1/2   Check 0/1/2 channel of the image for threshold
Return:
    mask:np array:  Binary numpy array of same shape as input image of threshold values
'''
@nb.njit
def threshMask(img,lowerbound=np.array([0,0,0]),upperbound=np.array([1,1,255]),mode:int=-1):    
    mask=np.full((img.shape[0],img.shape[1]),False)
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            if mode==-2:
                mask[i,j]=(img[i,j,:]>=lowerbound).all() and (img[i,j,:]<=upperbound).all()
            elif mode==-1:                
                mask[i,j]=(img[i,j,:]>=lowerbound).any() and (img[i,j,:]<=upperbound).any()  
            else:
                mask[i,j]=img[i,j,mode]>=lowerbound[mode] and img[i,j,mode]<=upperbound[mode]           
    return mask

"""
Center zoom in/out of the given image and returning an enlarged/shrinked view of the image without changing dimensions
------
Args:
    img : ndarray:  Image array
    zoom_factor : float:    amount of zoom as a ratio [0 to Inf). Default 0.
------
Returns:
    result: ndarray:    numpy ndarray of the same shape of the input img zoomed by the specified factor.          
"""
def cv2_clipped_zoom(img, zoom_factor=0):

    
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int32)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def augment(img,mask,path='BGseg/',rotate=True,zoom=True,bg=True,n=5):
    
    
    mask=np.array(mask,dtype=np.uint8)
    mask*=255
    
    imgs=[img]
    
    masks=[mask]
    for i in range(n):
        newimg=np.copy(img)
        newmask=np.copy(mask)
        
        if bg:
            image=[name for name in os.listdir(path)]
            image=image[random.randint(0,len(image)-1)]
            image=cv2_clipped_zoom(ndimage.rotate(cv2.resize(cv2.cvtColor(cv2.imread(path+'/'+image),cv2.COLOR_BGR2RGB),(img.shape[0],img.shape[1])), random.randint(0,360), reshape=False),(random.random()/2)+1)
            
            newimg=cv2.bitwise_and(newimg,newimg, mask=newmask)
            image=cv2.bitwise_and(image,image, mask=255 - newmask)
            
            newimg=cv2.add(newimg,image,dtype=cv2.CV_64F)
        if rotate:
            rran=random.randint(0,360)
            newimg=ndimage.rotate(newimg, rran, reshape=False)
            newmask=ndimage.rotate(newmask, rran, reshape=False)
        if zoom:
            zran=(random.random()/2)+1
            newimg=cv2_clipped_zoom(newimg,zran)
            newmask=cv2_clipped_zoom(newmask,zran)
        
        imgs.append(newimg.astype(np.uint8))
        masks.append(newmask.astype(np.uint8))
    return imgs,masks

'''
saveImage:  Save numpy mage to the given path
Args:
        img:ndarray    numpy image
        path:str       Save path
        name:Str       Save Name
'''
def saveImage(img,path:str,name:str,changeToBGR=True):
    #Check if path exists or creats it
    Path(path).mkdir(parents=True, exist_ok=True)
    #Write image to the path    
    if changeToBGR:
        cv2.imwrite(path+'\\'+name, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path+'\\'+name, img)
    
if __name__=='__main__':
    td=Data(info=True,path='data\\raw')
    td.changeColor=True
    count=0
    mask=[]
    imgs=[]
    def temp(img,relpath,name):
        global count,imgs,mask
        count+=1
        
        cm=(threshMask(rgb_to_hsv(img),lowerbound=np.array([0.02,0,0]),upperbound=np.array([0.6,0,0]),mode=0))
        ai,am=augment(img,cm,path='data/raw/Negative/Negative/',rotate=True,zoom=True,bg=True,n=5)
        imgs+=ai
        mask+=am
        
    td.proc({'func':temp,'randomBatchN':1,'loadImg':True})
    print(count)
    
    
   
    