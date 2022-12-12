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
import numpy as np
import numba as nb
from scipy import ndimage
import threading
import json

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
            Custom:     Returns anything tht the user wants (Default None)
            
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
        
        if 'return' in data and data['return']:
            return self.returnVal
           
    #Gets images from the folder with labels as immediated heirarchy folder name to the path  
    #func=None,tkinterStatus=None,randomBatchN:int=0,updateInfoEvery=10,loadImg=True,
    def balProc(self,data):        
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
                files=[]
                #For each label get all files from the subdirectories
                for ipath, subdirs, filest in os.walk(os.path.join(self.path, label)):
                    for filen in filest:
                        files.append([ipath,filen])
                    
                if 'randomBatchN' in data:
                    if len(files)>data['randomBatchN']:
                        random.shuffle(files)
                        files=files[:data['randomBatchN']] 
                
                
                for file in files:                          
                    if self.info:
                        if updateCheck>=updateInfoEvery:
                            status= file[0]+'/'+file[1]+' '+(str(round(updateInfoEvery/(time.time()-lt)))+'/s') if time.time()>lt else ''
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
                        img=getImage(os.path.join(file[0],file[1]),(self.x,self.y),self.changeColor)
                    
                    #Calling the process function
                    if 'func' in data:
                        data['func'](img,os.path.relpath(file[0], self.path),file[1])
                            
                        
                                                    
                        
            if 'tkinterStatus' in data:
                data['tkinterStatus'].set('Proccessed Images') 
                
            if 'onEnd' in data:
                data['onEnd']()
        
        except Exception :
            print('\nFailed to get Images\n',traceback.print_exc())
                        
        #Stop the loader
        if self.info:             
            loader.stop()
        
        if 'return' in data and data['return']:
            return self.returnVal
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


'''
augment:
    augments image and mask with different background to produce n images
    Args:
        img:ndarry
            img to be augmented
        mask:ndarry
            mask for the respective image needed to seperate foreground and replace background
        path:str
            path where the background images are stores, chooses a random image from the folder
        rotate:bool
            boolean check to rotate image while augmentation
        zoom:bool
            boolean check to zoom image while augmentation
        bg:bool
            boolean check to change background from the given path
        n:int
            number of augmented images that needed to be produced
'''
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
createPathIfNotExist:
    Check if path exists, if not then creats it
    Args:
        path:str    Path to be created
    
'''
def createPathIfNotExist(path:str):
    Path(path).mkdir(parents=True, exist_ok=True)

'''
saveImage:  
    Save numpy mage to the given path
    Args:
            img:ndarray    numpy image
            path:str       Save path
            name:Str       Save Name
'''
def saveImage(img,path:str,name:str,changeToBGR=True,isGray=False):
    #Check if path exists or creats it
    createPathIfNotExist(path)
    #Write image to the path    
    img=np.float32(img) 
    if isGray:               
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    if changeToBGR:
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path+'\\'+name, img)

'''
getImageFromFolderAsync:
    Using threading get Images from the folder with subfolders in async
    Args:
        XPath:str   path for features
        YPath:str   path for labels
        n:int       number of images to be taken per subfolder
'''
def getImageFromFolderAsync(XPath:str,YPath:str,n:int=5):
    
    #create data object for processing
    getData=Data()
    
    #setting the train folder path
    getData.path=XPath
    
    #variable to be returned in async
    getData.returnVal={}
    getData.returnVal['X']=[]
    getData.returnVal['Y']=[]
    
    #setting the class varibale for train and test paths
    getData.XPath=XPath
    getData.YPath=YPath
    
    
    #update function for each image
    def getimg(img,relpath,name):
        
        #Get image from the XPath
        getData.returnVal['X'].append(getImage(os.path.join(os.path.join(getData.XPath, relpath),name),(getData.x,getData.y),getData.changeColor))
        
        #Get image from the YPath
        getData.returnVal['Y'].append(getImage(os.path.join(os.path.join(getData.YPath, relpath),name),(getData.x,getData.y),getData.changeColor))        
       
    #Setting the thread    
    thread=threading.Thread(target=getData.proc,args=({'func':getimg,'return':True,'loadImg':True,'randomBatchN':n},))
    thread.setDaemon(True)
    thread.start() 
    
    #Returning the started thread to do the processing
    return thread,getData  

def getImageWithFolderIDAsync(XPath:str,n:int=5,doSwmad=False,useBalProc=False):
    #create data object for processing
    getData=Data()
    
    #setting the train folder path
    getData.path=XPath
    
    #variable to be returned in async
    getData.returnVal={}
    getData.returnVal['X']=[]
    getData.returnVal['Y']=[]
    
    
    folders=getFolders(XPath)
    
    def getimg(img,relpath,name):
        if doSwmad:
            img=np.concatenate((img,swmad(img)),axis=2)
        #Get image from the XPath
        getData.returnVal['X'].append(img)
        
        #Get onehot encoded folder name
        temp=np.zeros(len(folders))
        fname=relpath
        if '\\' in fname:
            fname=fname[:fname.index('\\')]
        temp[folders.index(fname)]=1
        
        getData.returnVal['Y'].append(temp)    

    #Setting the thread    
    if useBalProc:
        thread=threading.Thread(target=getData.balProc,args=({'func':getimg,'return':True,'loadImg':True,'randomBatchN':n},))
    else:
        thread=threading.Thread(target=getData.proc,args=({'func':getimg,'return':True,'loadImg':True,'randomBatchN':n},))
    thread.setDaemon(True)
    thread.start() 
    
    #Returning the started thread to do the processing
    return thread,getData 
'''
getFolders:
    get the names of the folder in the given path
    Args:
        path:str        The path in which the list of folder is needed
    Return:
        folders:list    The list of folder in string present in the path
'''
def getFolders(path):
    return sorted([f.name for f in os.scandir(path.strip()) if f.is_dir()])

'''
binaryProb:
    returns boolean for required probability
    Args:
        ratio:float
            the ratio at which binary threshold is needed
    Return:
        binary Prob:boolean
            gives a random binary number with the inputted ratio
'''
def binaryProb(ratio:float):
    return random.random()<ratio


'''
removeNoiseMask:
    removes noise from the mask like small holes or fills on the edges
    Args:
        mask:ndarry
            the raw mask on which the filter is needed
    Return:
        mask:ndarry
            the filtered mask after removing the noise
'''
def removeNoiseMask(mask):
    mask=np.float32(mask) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    return mask


'''
fillMask
'''
def fill_mask(mask):
    out=ndimage.binary_fill_holes(mask)
    return out.astype(np.int32())

'''
checkIfFileExist:
    CHecks if the file exists in the given list
    Args:
        paths:list
            the list of string which needs to be checked
    Return:
        exist?:bool
            if any on the path in the list exists, it returns True
'''
def checkIfFileExist(paths):
    for path in paths:
        if Path(path).is_file():
            return True
    return False


'''
swmad:
    generates sliding window mean absolute deviation
    Args:
        imgs:list
            list of images to be processed
        group_size:int
            size of the group for calculating mean 
        thresholde:int
            threshold value for rgb to be included while calculating
'''
@nb.njit(parallel=True)
def swmadnb(img,avg,group_size=10,threshold=10):
    #make zero array with same shape as images
    
    new=np.zeros_like(img)    
    
    for i in nb.prange(np.shape(img)[0]):        
        for j in nb.prange(np.shape(img)[1]):   
            
            #Check the rgb threshold
            if img[i,j,0]>threshold or img[i,j,1]>threshold or img[i,j,2]>threshold:
                try:
                    
                    new[i,j,:]=np.absolute(img[i,j,:]-avg[i,j,:])

                except:
                    pass 
    
              
    
    return new

def swmad(img,group_size=5,threshold=10):
    avg =  cv2.medianBlur(img,group_size*2+1)
    
    # return np.absolute(np.subtract(avg.astype(np.int),img.astype(np.int))).astype(np.uint8)
    return swmadnb(avg.astype(np.int),img.astype(np.int),group_size,threshold).astype(np.uint8)

'''
highlight:
    Highlights the specific value from 255
    Args:
        x:int
            the value of the pixel
        p:int
            the value of desired highlighted peak
        s:float
            spread the the function for highlight
'''
@nb.njit
def highlight(x,p,s=0.01):
    return (255)*s**((1/s)*(((x-p)**2)/((255**2)-((x-p)**2))))


'''
highlightColor:
    Hihlights a specfic rgb value in the image
    Args:
        img:ndarray
            numpy array containing the image
        rgb:ndarray
            numpy array containg the rgb value of hioghlight peak
        spread:ndarray
            numpy array containg the spread value for r g and b
        threshold:int
            threshold value for each pixel to be considered for highlight
'''
@nb.njit
def highlightColor(img,rgb=np.array([50,10,10]),spread=np.array([0.05,0.075,0.075]),threshold=10):
    
    img=img.astype(np.float32)
    rgb=rgb.astype(np.float32)
    std=np.zeros(img.shape)
    
    
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            if img[i,j,0]>threshold or img[i,j,1]>threshold or img[i,j,2]>threshold:
                img[i,j,0]=highlight(max(0.001,img[i,j,0]),rgb[0],spread[0])
                img[i,j,1]=highlight(max(0.001,img[i,j,1]),rgb[1],spread[1])
                img[i,j,2]=highlight(max(0.001,img[i,j,2]),rgb[2],spread[2])
                std[i,j,:]=np.std(img[i,j,:])
    # std/=np.max(std)
    # img*=(1-std)
    
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            img[i,j,:]=np.min(img[i,j,:])
    return img.astype(np.uint8)

'''
overlayHighlight:
    overlay image with the highlighted pixel value
    Args:
        img:ndarray
            numpy array containing the image to generate the highlight map
        overlay:ndarray
            numpy array containing the overlay image
        rgb:ndarray
            numpy array containing the highlight peak values in r g and b
        spread:ndarray
            numpy array containg the spread value for r g and b
        threshold:int
            threshold value for each pixel to be considered for highlight
'''
@nb.njit
def overlayHighlight(img,overlay,rgb=np.array([50,10,10]),spread=np.array([0.05,0.075,0.075]),threshold=10):
    img=highlightColor(img,rgb=rgb,spread=spread,threshold=threshold).astype(np.float32)
    img/=np.max(img)
    return (overlay.astype(np.float32)*(1-img)).astype(np.uint8)

def overlayMask(img,mask,is255=True,to255=False):
    mask=np.reshape(mask,(mask.shape[0],mask.shape[1],1))   
    mask=np.array(mask,dtype=np.float32)
    if is255:
        mask/=255
    mask=mask*img
    if to255:
        mask=mask * 255
    return mask.astype(np.uint8)
    

def calculateDiseasePart(img,plant:str="",returnProcImg:bool=False):
    imgh=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if plant=="":
        imgh=overlayHighlight(imgh,img,rgb=np.array([60,170,100]),spread=np.array([0.05,0.99,0.99])).astype(np.float32)
        imgg=overlayHighlight(img,imgh,rgb=np.array([150,150,150]),spread=np.array([0.3,0.3,0.3])).astype(np.float32)
    else:
        f = open('data/healthyHSV.json')
        vals=json.load(f)
        
        rgbhp=np.array([vals[plant]['Healthy']['ph'],vals[plant]['Healthy']['ps'],vals[plant]['Healthy']['pv']])
        rgbhs=np.array([vals[plant]['Healthy']['sh'],vals[plant]['Healthy']['ss'],vals[plant]['Healthy']['sv']])
        rgbbp=np.array([vals[plant]['Background']['ph'],vals[plant]['Background']['ps'],vals[plant]['Background']['pv']])
        rgbbs=np.array([vals[plant]['Background']['sh'],vals[plant]['Background']['ss'],vals[plant]['Background']['sv']])
        f.close()
        imgh=overlayHighlight(imgh,img,rgb=rgbhp,spread=rgbhs).astype(np.float32)
        imgg=overlayHighlight(cv2.cvtColor(img, cv2.COLOR_RGB2HSV),imgh,rgb=rgbbp,spread=rgbbs).astype(np.float32)
    
    if returnProcImg:
        return round((np.sum(imgg)/np.sum(img))*100,3),imgg
    else:
        return round((np.sum(imgg)/np.sum(img))*100,3)
    

                
def bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def rgb_to_hsv(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

if __name__=='__main__':
    
    img=getImage('data\\train\\segment\\4\\Apple\\Apple_scab\\image (13).jpg',changeColor=True)
    
    


    
 
   
    