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
    
    def __init__(self,datasetPath:str='data/train/manseg',setModel=True):
        self.path=datasetPath
        if setModel:
            self.setModel()
            
    '''
    setModel:
        to set the model into the class variable
        Args:
            input shape:
                Tuple containing the input shape of the image
        Return:
            None
    '''
    def setModel(self,input_shape=(256,256,3)):
        self.model = tf.keras.models.Sequential()            
        
        self.model.add(tf.keras.layers.Convolution2D(16, (2,2), padding='same',input_shape=input_shape))        
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        
        self.model.add(tf.keras.layers.Convolution2D(16, (3,3), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.model.add(tf.keras.layers.Convolution2D(32, (2,2), padding='same'))        
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))        
        
        self.model.add(tf.keras.layers.Convolution2D(32, (3,3), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))        
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.model.add(tf.keras.layers.Convolution2D(64, (2,2), padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.model.add(tf.keras.layers.Convolution2D(64, (3,3), padding='same'))

        
        self.model.add(tf.keras.layers.Convolution2D(64, (5,5), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.model.add(tf.keras.layers.Convolution2D(128, (2,2), padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.model.add(tf.keras.layers.Convolution2D(128, (3,3), padding='same'))
        
        self.model.add(tf.keras.layers.Convolution2D(128, (5,5), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.model.add(tf.keras.layers.Convolution2D(256, (2,2), padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.model.add(tf.keras.layers.Convolution2D(256, (3,3), padding='same'))

        
        self.model.add(tf.keras.layers.Convolution2D(256, (5,5), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="same"))
        
        self.model.add(tf.keras.layers.Convolution2D(512, (2,2), padding='same'))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        
        self.model.add(tf.keras.layers.Convolution2D(512, (3,3), padding='same'))

        
        self.model.add(tf.keras.layers.Convolution2D(512, (5,5), padding='same'))
        
        self.model.add(tf.keras.layers.Convolution2D(512, (7,7), padding='same'))
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.Conv2DTranspose(512, (2,2), padding='same'))    
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.model.add(tf.keras.layers.Conv2DTranspose(256, (2,2), padding='same'))    
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.model.add(tf.keras.layers.Conv2DTranspose(128, (2,2), padding='same'))    
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.model.add(tf.keras.layers.Conv2DTranspose(64, (2,2), padding='same'))    
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        
        self.model.add(tf.keras.layers.Conv2DTranspose(32, (2,2), padding='same'))    
        self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(tf.keras.layers.BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(tf.keras.layers.Dropout(0.3))
        
        self.model.add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        
        self.model.add(tf.keras.layers.Conv2DTranspose(1, (2,2), padding='same',activation='sigmoid'))            
    '''
    compileModel:
        compiles the setted model
        Args:
            None
        Return:
            None
    '''
    def compileModel(self):
        self.model.compile(optimizer='Nadam', loss="mean_absolute_error", metrics=['accuracy'])
        
    
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
    def fitModel(self,x_train,y_train,x_test,y_test,batch_size:int=10,epochs=25):
        self.history=self.model.fit(x=x_train,y=y_train,validation_data = (x_test,y_test),epochs = epochs,workers=6,batch_size=batch_size)
        
    def preloadData(self,XTrainPath,YTrainPath,XTestPath,YTestPath,n=5):
        self.preLoadTrainThread,self.preLoadTrainData=iproc.getImageFromFolderAsync(XTrainPath,YTrainPath,n)        
        self.preLoadTestThread,self.preLoadTestData=iproc.getImageFromFolderAsync(XTestPath,YTestPath,n)
        
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
        
    def trainBatchWise(self,batches=10):
        self.preloadData('data/train/manseg/image/Tomato','data/train/manseg/mask/Tomato','data/test/manseg/image/Tomato','data/test/manseg/mask/Tomato',5)        
        for i in range(batches):
            self.loadFromPreload()
            self.preloadData('data/train/manseg/image/Tomato','data/train/manseg/mask/Tomato','data/test/manseg/image/Tomato','data/test/manseg/mask/Tomato',5)      
            
            self.fitModel(self.x_train, self.y_train, self.x_test, self.y_test)
 
def custom_loss(y_true, y_pred): 
    y_true=tf.cast(y_true, tf.float32)    
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(tf.subtract(y_true,y_pred)),1),1),1),0)

if __name__=='__main__':
    test=SegmentML()
    test.setModel()
    test.compileModel()
    test.trainBatchWise()
    
    
    