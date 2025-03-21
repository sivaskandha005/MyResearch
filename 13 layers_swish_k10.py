# -*- coding: utf-8 -*-

#Part 1 : Building a CNN

#import Keras packages
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
import random
import datetime
from math import exp
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.backend import sigmoid
#import matplotlib.pyplot as plt
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})
#print(swish(2))
nb_train_samples = 346                # Number of train samples
nb_validation_samples = 340            # Number of validation samples

# Initializing the CNN

#Part 2 - fitting the data set

import glob
healthy = glob.glob('./healthy/*.*')
malign = glob.glob('./malign/*.*')

data = []
labels = []

for i in healthy:
    image=load_img(i, 
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(0)
    
for i in malign:   
    image=load_img(i,  
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(1)


data = np.array(data)
labels = np.array(labels)
#print(len(labels))
from keras.utils.np_utils import to_categorical   

categorical_labels = to_categorical(labels, num_classes=2)


from sklearn.model_selection import train_test_split
start_time=datetime.datetime.now()
#X_train1, X_test0, ytrain1, ytest0 = train_test_split(data, categorical_labels, test_size=0.1,
                                                    #random_state=random.randint(0,100))
X_train1 = data
ytrain1 = categorical_labels


for index in range(10):
    np.random.seed(1337)
    

    classifier = Sequential()

    classifier.add(Convolution2D(64, kernel_size=(3, 3),strides=(1,1), input_shape = (256, 256, 3),padding='SAME', activation = 'swish'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32,kernel_size=(3, 3),strides=(1,1), activation = 'swish',padding='same'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(16, kernel_size=(3, 3), activation = 'swish',padding='same'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(8, kernel_size=(3, 3), activation = 'swish',padding='same'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(16, kernel_size=(3, 3), activation = 'relu',padding='same'))
    classifier.add(AveragePooling2D(pool_size = (2, 2)))
    
    
    classifier.add(Flatten())

#hidden layer
    classifier.add(Dense(output_dim = 128, activation = 'swish'))
    classifier.add(Dropout(p = 0.5))
    
    #output layer
    classifier.add(Dense(output_dim = 2, activation = 'softmax'))
    
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



    X_train, X_test1, ytrain, ytest1 = train_test_split(X_train1, ytrain1, test_size=0.1,
                                                    random_state=random.randint(0,100))
    
    X_val, X_test, yval, ytest = train_test_split(X_test1, ytest1, test_size=0.5,
                                                    random_state=random.randint(0,100))
    print(len(X_test1))
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow(
            X_train, ytrain,
            batch_size=32)
    
    val_set = val_datagen.flow(
            X_val, yval,
            batch_size=32)
    
    test_set = test_datagen.flow(
            X_test1, ytest1,
            batch_size=120)
    #print(test_set)
    X, y = test_set.next()

    history=classifier.fit_generator(
            training_set,
            steps_per_epoch=20,
            epochs=50,
            validation_data=val_set,verbose=2,
            validation_steps=100)
    print("******History**** {}".format(index))
    print(history.history['acc'],history.history['val_acc'],history.history['loss'],history.history['val_loss'])     
    print("******end*******")
    classifier.save_weights('AFndAS_trained_model_weights_11layers_{}_k10.h5'.format(index))
    #plt.plot(history.history['acc'],history.history['loss'])
    #plt.plot(history.history['val_acc'],history.history['val_loss'])
    
    arr = classifier.evaluate(X,y)
    print(arr)
    arr = classifier.predict_classes(X)
    print(arr)
    print(y)
end_time=datetime.datetime.now()
print("Time duration forcomplete cyle is",end_time- start_time)
