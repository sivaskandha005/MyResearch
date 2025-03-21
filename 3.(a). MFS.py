import numpy as np
from keras import models
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import Activation
from keras.regularizers import l2
import cv2
import os
from keras.utils import get_file
from keras import applications

train = "D:/Research Work/26-02-2020/AF and AS 1/AF and AS 1_10 Cross/K2 cros/case2 - black/Case 1/train/AS/"
#test = "D:/Research Work/26-02-2020/AF and AS 1/AF and AS 1_10 Cross/masking/case 2/test2_AS/"
model=applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))


model = Sequential()

model.add(Convolution2D(256, kernel_size=(3, 3),strides=(1,1), input_shape = (240, 240, 3),padding='SAME', activation = 'relu'))

model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Convolution2D(128,kernel_size=(3, 3),strides=(1,1), activation = 'relu',padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Convolution2D(64, kernel_size=(3, 3), activation = 'relu',padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Convolution2D(32, kernel_size=(3, 3), activation = 'relu',padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(Convolution2D(16, kernel_size=(3, 3), activation = 'relu',padding='same'))
model.add(AveragePooling2D(pool_size = (2, 2)))

model.add(Flatten())

#hidden layer
model.add(Dense(output_dim = 128, activation = 'relu',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
model.add(Dropout(p = 0.5))

#output layer
model.add(Dense(output_dim = 2, activation = 'softmax'))

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.load_weights('keras_AFndAS_trained_model_case2_run1_softmax_weightsregularized_Avgpooling_10000.h5')

for i in os.listdir(train):
    if i.endswith('jpg'):
        print(i)

        img_path = train + i
        img = image.load_img(img_path, target_size=(240, 240))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        #plt.imshow(img_tensor[0])
        #plt.show()
        #print(img_tensor.shape)
        
        
        from keras import backend as K
        
        layer_outputs = [layer.output for layer in model.layers[:13]] 
        # Extracts the outputs of the top 12 layers
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
        
        activations = activation_model.predict(img_tensor) 
#print(len(activations[9][0]))



     

k1=activations[0].reshape(-1)
k2=activations[1].reshape(-1)
k3=activations[2].reshape(-1)
k4=activations[3].reshape(-1)
k5=activations[4].reshape(-1)
k6=activations[5].reshape(-1)
k7=activations[6].reshape(-1)
k8=activations[7].reshape(-1)
k9=activations[8].reshape(-1)
k10=activations[9].reshape(-1)
k11=activations[10][0]
k12=activations[11][0]
print("layer1", sum(k1)/len(k1))
print("layer2", sum(k2)/len(k2))
print("layer3", sum(k3)/len(k3))
print("layer4", sum(k4)/len(k4))
print("layer5", sum(k5)/len(k5))
print("layer6", sum(k6)/len(k6))
print("layer7", sum(k7)/len(k7))
print("layer8", sum(k8)/len(k8))
print("layer9", sum(k9)/len(k9))
print("layer10", sum(k10)/len(k10))
print("layer11", sum(k11)/len(k11))
print("layer12", sum(k12)/len(k12))



"""       
        first_layer_activation = activations[0]
        #print(first_layer_activation.shape)
        plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
        
        layer_names = []
        for layer in model.layers[:12]:
            layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
         
        print(layer_names)
        images_per_row = 1
        
        index = 1
        for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
            print(layer_name)
            n_features = layer_activation.shape[-1] # Number of features in the feature map
            size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
            n_cols = 1 # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                                     :, :,
                                                     col * images_per_row + row]
                    #print(channel_image)
                    channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size : (col + 1) * size, # Displays the grid
                                 row * size : (row + 1) * size] = channel_image
                    print(display_grid)
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            #img_path1 = test + "Layer " + str(index) + "/"  + layer_name + i
            print(img_path1)
            #plt.imsave(img_path1, display_grid)
            index += 1
"""
