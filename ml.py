# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:27:22 2022

@author: vedhs
"""
from keras.self.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LeakyReLU
import tensorflow as tf


'''
segmentML:
    To segment images and generate a mask from it using Conv and ConvTranspose layers
    Args:
        None
'''
class segmentML:
    
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
        self.model = Sequential()    
    
        self.model.add(Convolution2D(64, (3,3), padding='same',input_shape=input_shape))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
        
        self.model.add(Convolution2D(64, (3,3),strides=(2,2), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
        
        self.model.add(Convolution2D(128, (3,3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
        
        self.model.add(Convolution2D(128, (3,3),strides=(2,2), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
       
        self.model.add(Convolution2D(256, (3,3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
        
        
        self.model.add(Conv2DTranspose(128, (3,3),strides=(2,2), padding='same'))    
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
        
        self.model.add(Convolution2D(128, (3,3), padding='same'))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
        
        self.model.add(Conv2DTranspose(64, (3,3),strides=(2,2), padding='same'))    
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(BatchNormalization(momentum=0.15, axis=-1))
        self.model.add(Dropout(0.3))
        
        
        self.model.add(Convolution2D(1, (3,3), padding='same',activation='sigmoid')) 
       
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
    def fitModel(self,x_train,y_train,x_test,y_test,batch_size:int=10):
        self.history=self.model.fit(x=x_train,y=y_train,validation_data = (x_test,y_test),epochs = 100,workers=6,batch_size=batch_size)
        
def custom_loss(y_true, y_pred): 
    y_true=tf.cast(y_true, tf.float32)    
    return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(tf.subtract(y_true,y_pred)),1),1),1),0)