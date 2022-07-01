# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:27:22 2022

@author: vedhs
"""
import tensorflow as tf
import iproc
import numpy as np
'''
segmentML:
    To segment images and generate a mask from it using Conv and ConvTranspose layers
    Args:
        None
'''
class SegmentML:
    
    def __init__(self,datasetPath:str='data/train/manseg'):
        self.path=datasetPath
        self.setModel={}
        self.setModel[1]=self.__setModel1
        self.setModel[2]=self.__setModel2
        self.setModel[3]=self.__setModel3
        self.models={}
        self.modelSavePath=''
        for model in self.setModel:
            self.setModel[model]()
            
    
        
    
    '''
    __setModel1:
        to set model with type 1 structure
        Args:
            input shape:
                Tuple containing the input shape of the image
        Return:
            keras seq model
            
    __setModel2:
        to set model with type 2 structure
        Args:
            input shape:
                Tuple containing the input shape of the image
        Return:
            keras seq model
            
    __setModel3:
        to set model with type 3 structure
        Args:
            input shape:
                Tuple containing the input shape of the image
        Return:
            keras seq model
            
    '''
    def __setModel1(self,input_shape=(256,256,3)):
        self.models[1] = tf.keras.models.Sequential()            
        
        self.models[1].add(tf.keras.layers.Convolution2D(16, (3,3), padding='same',input_shape=input_shape))
        self.models[1].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[1].add(tf.keras.layers.Convolution2D(16, (2,2), padding='same'))
        self.models[1].add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.models[1].add(tf.keras.layers.Dropout(0.3))
        
        self.models[1].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[1].add(tf.keras.layers.Convolution2D(32, (5,5), padding='same'))        
        self.models[1].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))  
        
        self.models[1].add(tf.keras.layers.Convolution2D(32, (3,3), padding='same'))      
        
        self.models[1].add(tf.keras.layers.Convolution2D(32, (2,2), padding='same'))
        self.models[1].add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.models[1].add(tf.keras.layers.Dropout(0.3))
        
        self.models[1].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[1].add(tf.keras.layers.Convolution2D(64, (7,7), padding='same'))
        self.models[1].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[1].add(tf.keras.layers.Convolution2D(64, (5,5), padding='same'))
        
        self.models[1].add(tf.keras.layers.Convolution2D(64, (3,3), padding='same'))

        self.models[1].add(tf.keras.layers.Convolution2D(64, (2,2), padding='same'))
        self.models[1].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[1].add(tf.keras.layers.Dropout(0.3))
        
        self.models[1].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[1].add(tf.keras.layers.Convolution2D(128, (9,9), padding='same'))
        self.models[1].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[1].add(tf.keras.layers.Convolution2D(128, (7,7), padding='same'))
        
        self.models[1].add(tf.keras.layers.Convolution2D(128, (5,5), padding='same'))
        
        self.models[1].add(tf.keras.layers.Convolution2D(128, (3,3), padding='same'))
        
        
        self.models[1].add(tf.keras.layers.Convolution2D(128, (2,2), padding='same'))
        self.models[1].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[1].add(tf.keras.layers.Dropout(0.3))
        
       
        
        self.models[1].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.models[1].add(tf.keras.layers.Conv2DTranspose(64, (2,2), padding='same'))    
        self.models[1].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[1].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.models[1].add(tf.keras.layers.Dropout(0.3))
        
        self.models[1].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        
        self.models[1].add(tf.keras.layers.Conv2DTranspose(32, (2,2), padding='same'))    
        self.models[1].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[1].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.models[1].add(tf.keras.layers.Dropout(0.3))
        
        self.models[1].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.models[1].add(tf.keras.layers.Conv2DTranspose(1, (2,2), padding='same',activation='sigmoid'))            

    def __setModel2(self,input_shape=(256,256,3)):
        self.models[2] = tf.keras.models.Sequential()            
        
        self.models[2].add(tf.keras.layers.Convolution2D(16, (3,3), padding='same',input_shape=input_shape))
        self.models[2].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[2].add(tf.keras.layers.Convolution2D(16, (2,2), padding='same'))
        self.models[2].add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.models[2].add(tf.keras.layers.Dropout(0.3))
        
        self.models[2].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[2].add(tf.keras.layers.Convolution2D(32, (5,5), padding='same'))        
        self.models[2].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))  
        
        self.models[2].add(tf.keras.layers.Convolution2D(32, (3,3), padding='same'))      
        
        self.models[2].add(tf.keras.layers.Convolution2D(32, (2,2), padding='same'))
        self.models[2].add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.models[2].add(tf.keras.layers.Dropout(0.3))
        
        self.models[2].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[2].add(tf.keras.layers.Convolution2D(64, (7,7), padding='same'))
        self.models[2].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[2].add(tf.keras.layers.Convolution2D(64, (5,5), padding='same'))
        
        self.models[2].add(tf.keras.layers.Convolution2D(64, (3,3), padding='same'))

        self.models[2].add(tf.keras.layers.Convolution2D(64, (2,2), padding='same'))
        self.models[2].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[2].add(tf.keras.layers.Dropout(0.3))
        
        self.models[2].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[2].add(tf.keras.layers.Convolution2D(128, (9,9), padding='same'))
        self.models[2].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[2].add(tf.keras.layers.Convolution2D(128, (7,7), padding='same'))
        
        self.models[2].add(tf.keras.layers.Convolution2D(128, (5,5), padding='same'))
        
        self.models[2].add(tf.keras.layers.Convolution2D(128, (3,3), padding='same'))
        
        
        self.models[2].add(tf.keras.layers.Convolution2D(128, (2,2), padding='same'))
        self.models[2].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[2].add(tf.keras.layers.Dropout(0.3))
        
       
        
        self.models[2].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.models[2].add(tf.keras.layers.Conv2DTranspose(64, (2,2), padding='same'))    
        self.models[2].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[2].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.models[2].add(tf.keras.layers.Dropout(0.3))
        
        self.models[2].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        
        self.models[2].add(tf.keras.layers.Conv2DTranspose(32, (2,2), padding='same'))    
        self.models[2].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[2].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.models[2].add(tf.keras.layers.Dropout(0.3))
        
        self.models[2].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.models[2].add(tf.keras.layers.Conv2DTranspose(1, (2,2), padding='same',activation='sigmoid'))                        
        
    def __setModel3(self,input_shape=(256,256,3)):
        self.models[3] = tf.keras.models.Sequential()            
        
        self.models[3].add(tf.keras.layers.Convolution2D(16, (3,3), padding='same',input_shape=input_shape))
        self.models[3].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[3].add(tf.keras.layers.Convolution2D(16, (2,2), padding='same'))
        self.models[3].add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.models[3].add(tf.keras.layers.Dropout(0.3))
        
        self.models[3].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[3].add(tf.keras.layers.Convolution2D(32, (5,5), padding='same'))        
        self.models[3].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))  
        
        self.models[3].add(tf.keras.layers.Convolution2D(32, (3,3), padding='same'))      
        
        self.models[3].add(tf.keras.layers.Convolution2D(32, (2,2), padding='same'))
        self.models[3].add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.models[3].add(tf.keras.layers.Dropout(0.3))
        
        self.models[3].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[3].add(tf.keras.layers.Convolution2D(64, (7,7), padding='same'))
        self.models[3].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[3].add(tf.keras.layers.Convolution2D(64, (5,5), padding='same'))
        
        self.models[3].add(tf.keras.layers.Convolution2D(64, (3,3), padding='same'))

        self.models[3].add(tf.keras.layers.Convolution2D(64, (2,2), padding='same'))
        self.models[3].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[3].add(tf.keras.layers.Dropout(0.3))
        
        self.models[3].add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.models[3].add(tf.keras.layers.Convolution2D(128, (9,9), padding='same'))
        self.models[3].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.models[3].add(tf.keras.layers.Convolution2D(128, (7,7), padding='same'))
        
        self.models[3].add(tf.keras.layers.Convolution2D(128, (5,5), padding='same'))
        
        self.models[3].add(tf.keras.layers.Convolution2D(128, (3,3), padding='same'))
        
        
        self.models[3].add(tf.keras.layers.Convolution2D(128, (2,2), padding='same'))
        self.models[3].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[3].add(tf.keras.layers.Dropout(0.3))
        
       
        
        self.models[3].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.models[3].add(tf.keras.layers.Conv2DTranspose(64, (2,2), padding='same'))    
        self.models[3].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[3].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.models[3].add(tf.keras.layers.Dropout(0.3))
        
        self.models[3].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        
        self.models[3].add(tf.keras.layers.Conv2DTranspose(32, (2,2), padding='same'))    
        self.models[3].add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.models[3].add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.models[3].add(tf.keras.layers.Dropout(0.3))
        
        self.models[3].add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.models[3].add(tf.keras.layers.Conv2DTranspose(1, (2,2), padding='same',activation='sigmoid'))            
    '''
    compileModel:
        compiles the setted model
        Args:
            None
        Return:
            None
    '''
    def compileModel(self):
        for model in self.models:
            self.models[model].compile(optimizer='Nadam', loss="mean_absolute_error", metrics=['accuracy'])        
        
    
    '''
    fitModel:
        fits the model for the training and saves history in class variable
        Args:
            x_train:
                training features
            y_train:
                training labels
            x_test:
                test features
            y_test:
                test labels
            batch_size:int
                batch size for training
    '''
    def fitModel(self,x_train,y_train,x_test,y_test,model,batch_size:int=10,epochs=25):
        history = self.models[model].fit(x=x_train,y=y_train,validation_data = (x_test,y_test),epochs = epochs,workers=6,batch_size=batch_size)
        return history.history['accuracy'],history.history['val_accuracy']
        
    
        
            
        
    '''
    preloadData:
        preloads data in the background in parallel using threading
        Args:
            xTrainPath:str
                training path for X Train set
            YTrainPath:str
                training path for Y Train set
            XTestPath:str
                training path for X Test set
            YTestPath:str
                training path for Y Test set
            n:int
                number of random images per folder to take in the set
    '''
    def preloadData(self,XTrainPath:str,YTrainPath:str,XTestPath:str,YTestPath:str,n:int=5):        
        self.preLoadTrainThread,self.preLoadTrainData=iproc.getImageFromFolderAsync(XTrainPath,YTrainPath,n)        
        self.preLoadTestThread,self.preLoadTestData=iproc.getImageFromFolderAsync(XTestPath,YTestPath,n)
        
    '''
    loadFromPreLoad:
        waits for data to be preloaded and then loads data into numpy array suitable for model training
        Args:
            None
    '''
    def loadFromPreload(self):
        self.preLoadTrainThread.join()
        self.preLoadTestThread.join()
        
        
        self.x_train=np.array(self.preLoadTrainData.returnVal['X'],dtype=np.float32)
        self.y_train=np.array(self.preLoadTrainData.returnVal['Y'],dtype=np.float32)[:,:,:,0]
        self.y_train=np.reshape(self.y_train,(self.y_train.shape[0],self.y_train.shape[1],self.y_train.shape[2],1))
        self.x_test=np.array(self.preLoadTestData.returnVal['X'],dtype=np.float32)
        self.y_test=np.array(self.preLoadTestData.returnVal['Y'],dtype=np.float32)[:,:,:,0]
        self.y_test=np.reshape(self.y_test,(self.y_test.shape[0],self.y_test.shape[1],self.y_test.shape[2],1))
        
        self.y_train/=255
        self.y_test/=255
        
    '''
    trainBatchWise:
        train model batch wise, use this for huge data which cannot be simulataneously loaded into the batch
        Args:
            XTrainPath:str
                path for training features
            YTrainPath:str
                path for training labels
            XTestPath:str
                path for testing features
            YTestPath:str
                path for testing labels
            batches:int=10
                total random batches to be trained
            imagePerFolder:int=5
                images per folder to be taken
            epochs:int=10
                epochs per batch for training
            saveAfterTrain:bool=True
                Boolean check to save model after training
            deleteAfterTrain:bool=True
                Boolean check to delete the model after training and saving
        Return:
            acs:list
                list containing accuracy for each batch
            vacs:list
                list comntaing validation accuracy for each batch
            
    '''
    def trainBatchWise(self,XTrainPath:str,YTrainPath:str,XTestPath:str,YTestPath:str,batches:int=10,imagePerFolder:int=5,epochs:int=10,saveAfterTrain:bool=True,deleteAfterTrain:bool=True):
        self.preloadData(XTrainPath,YTrainPath,XTestPath,YTestPath,imagePerFolder)
        mac=[]
        macv=[]
        for model in self.models:            
            print('Training model',model)  
            acs=[]
            vacs=[]
            for i in range(batches):
                print('Batch',i+1,'of',batches)
                self.loadFromPreload()
                self.preloadData(XTrainPath,YTrainPath,XTestPath,YTestPath,imagePerFolder)                      
                ac,vac=self.fitModel(self.x_train, self.y_train, self.x_test, self.y_test,model,10,epochs)
                acs.append(ac)
                vacs.append(vac)
            if saveAfterTrain and self.modelSavePath != '':
                iproc.createPathIfNotExist(self.modelSavePath)
                self.models[model].save(self.modelSavePath+'/'+str(model)+'.h5')
            
            if deleteAfterTrain:
                self.models[model]=None
            mac.append(acs)
            macv.append(vacs)
        return mac,macv
 
    '''
    loadModels:
        loads the saved model
        Args:
            path:str
                path where the models are saved (directory)
        Return:
            None
    '''
    def loadModels(self,path):
        model_names=[1,2,3]
        for i in model_names:
            self.models[i]=tf.keras.models.load_model(path+'/'+str(i)+'.h5')
        
    '''
    predict:
        predicts the mask from the given image
        Args:
            x:ndarry
                numpy array of the image
        Return:
            y:ndarry
                numppy array of the mask
    '''
    def predict(self,x,smooth=True,fill=True):
        model_names=[1,2,3]
        if len(x.shape)==3:
            x=np.reshape(x,(1,x.shape[0],x.shape[1],x.shape[2]))
            
        y=[np.zeros((x.shape[1],x.shape[2]),dtype=np.float32)]*4
        yi=0
        for i in model_names:
            out=self.models[i].predict(x)[0,:,:,0]
            if smooth:
                out=iproc.removeNoiseMask(out)
            
            y[3]+=out
            out[out>=0.5]=255
            out[out<0.5]=0
            if fill:
                out/=255
                out=iproc.fill_mask(out)
                out*=255
            # out*=255
            y[yi]=out
            yi+=1
            
        y[3]/=len(model_names)
        y[3][y[3]>=0.5]=255
        y[3][y[3]<0.5]=0
        if fill:
            y[3]/=255
            y[3]=iproc.fill_mask(y[3])
            y[3]*=255
        # y[3]*=255
        for i in range(len(y)):
            y[i]=np.array(y[i],dtype=np.uint8)
        return y
    
    def predictAndSaveFolder(self,path:str,modelPath:str,savePath:str,subFolders=True,segment:bool=True,saveAll=True):
        folders=['']
        if subFolders:
            folders=iproc.getFolders(path)
        for folder in folders:
            
            pathn=path
            if subFolders:
                pathn+='\\'+folder
            
            modelPathn=modelPath
            if subFolders:
                modelPathn+='\\'+folder
            
            
                
            imgs=iproc.Data(path=pathn)
            
            self.loadModels(modelPathn)
            
            def getimg(img,relpath,name):
                              
                y=self.predict(img)
                if segment:
                    for i in range(len(y)):
                        y[i]=np.reshape(y[i],(y[i].shape[0],y[i].shape[1],1))   
                        y[i]=np.array(y[i],dtype=np.float32)
                        y[i]/=255
                        nimg=img*y[i]
                        y[i]=nimg
                for i in range(len(y)):
                    if saveAll:
                        if subFolders:
                            iproc.saveImage(y[i],savePath+'\\'+str(i+1)+'\\'+folder+'\\'+relpath,name)
                        else:
                            iproc.saveImage(y[i],savePath+'\\'+str(i+1)+'\\'+relpath,name)
                    else:
                        if i==3:
                            if subFolders:
                                iproc.saveImage(y[i],savePath+'\\'+folder+'\\'+relpath+'\\',name)    
                            else:
                                iproc.saveImage(y[i],savePath+'\\'+relpath+'\\',name)    
                
            imgs.proc({'func':getimg,'loadImg':True})
    
if __name__=='__main__':
    test=SegmentML()
    test.predictAndSaveFolder('data/train/raw', 'model/segment', 'data/train/segment')
    # print('You sucesssfully ran the print command and those imports')    
    
    
    