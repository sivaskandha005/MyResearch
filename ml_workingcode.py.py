# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018

@author: mohit123
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:17:40 2017

@author: Mohit
"""

#Part 1 : Building a CNN

#import Keras packages
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img

nb_train_samples = 2543                # Number of train samples
nb_validation_samples = 2261            # Number of validation samples

# Initializing the CNN
classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(8, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Flatten())

#hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(p = 0.5))

#output layer
classifier.add(Dense(output_dim = 2, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



#plt.show()
#print(j)
    
##print(arr)
#print(len(scores))
#print(scores)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import mahotas
import tensorflow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# fixed-sizes for image
fixed_size = tuple((256, 256))

# path to training data
train_path = "DataSet-faces95"

# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

def calculatehistogram(image, eps=1e-7):
    lbp = local_binary_pattern(image, 16, 2, method="uniform")
    (histogram, _) = np.histogram(lbp.ravel(),
                                  bins=np.arange(0, 16 + 3),
                                  range=(0, 16 + 2))
    #now we need to normalise the histogram so that the total sum=1
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + eps)
    return histogram 
    
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    #print(len(feature))
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    #print(len(haralick))
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    #print(len(hist.flatten()))
    return hist.flatten()

def fd_LBP(image):
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hist = calculatehistogram(gray)
    #kp = np.array(kp)
    #print(len(pts.flatten()))
    #print(len(hist.flatten()))
    return hist.flatten()

def fd_HOG(image):
    hog = cv2.HOGDescriptor()
    #im = cv2.imread(sample)
    image = cv2.resize(image, (128,128))
    h = hog.compute(image)
    return h.flatten()


# create all the machine learning models
models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9)))
models.append(('KNN', KNeighborsClassifier()))

models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('DT', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"


print( "[STATUS] training started...")

# split the training and testing data
"""
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)
"""
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 80

index_label = 0
sum_features1 = 0
sum_features2 = 0
len_feat = 0
# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    k = 1
    
    index_img = 0
    # loop over the images in each sub-folder
    for fileName in os.listdir(dir):
        # get the image file name
        file = dir + "\\" + str(fileName)

        #print(file)
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        #image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        #fv_LBP = fd_LBP(image)
        #fv_HOG = fd_HOG(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_haralick, fv_hu_moments, fv_histogram])

        len_feat = len(global_feature)
        if(index_label == 0 and index_img == 0):
            sum_features1 = global_feature
        elif(index_label == 0 and index_img > 0):
            sum_features1 += global_feature
        if(index_label == 1 and index_img == 0):
            sum_features2 = global_feature
        elif(index_label == 1 and index_img > 0):
            sum_features2 += global_feature
        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
        index_img += 1
        
    index_label += 1
    tot_img = index_img
    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1

#print(sum_features1)

mean_feature1 = sum_features1/tot_img
mean_feature2 = sum_features2/tot_img

feat_num = []
idx = 1
while idx <= len(mean_feature1):
    idx += 1
    feat_num.append(idx)

def plot_bar_x(feature, feat_num):
    # this is for plotting purpose
    index = np.arange(len(feat_num))
    plt.bar(index, feature)
    plt.xlabel('Feature number', fontsize=5)
    plt.ylabel('Feature Value', fontsize=5)
    plt.xticks(index, mean_feature1, fontsize=5, rotation=30)
    plt.title('Visualization of mean feature vector (HOG)')
    plt.show()

plot_bar_x(mean_feature1, feat_num)
plot_bar_x(mean_feature2, feat_num)
#print(mean_feature1)

from scipy.spatial import distance

#d = distance.euclidean(mean_feature1, mean_feature2)
#d = np.linalg.norm(mean_feature1- mean_feature2)
d = np.sqrt(np.sum((mean_feature1- mean_feature2) ** 2, axis=0))

print(d/len_feat)

print ("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print ("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] training labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print ("[STATUS] feature vector normalized...")

# save the feature vector using HDF5
h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

rescaled_features = np.array(global_features_string)
target = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()


trainDataGlobal=np.array(rescaled_features)
trainLabelsGlobal=np.array(target)

########################
"""
test_labels = os.listdir("val")

# sort the testing labels
test_labels.sort()
print(test_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 80

# loop over the testing data sub-folders
for testing_name in test_labels:
    # join the testing data path and each species testing folder
    dir = os.path.join("val", testing_name)

    # get the current testing label
    current_label = testing_name

    k = 1
    # loop over the images in each sub-folder
    for fileName in os.listdir(dir):
        # get the image file name
        file = dir + "\\" + str(fileName)

        #print(file)
        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        #image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1

print ("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall testing label size
print ("[STATUS] testing Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] testing labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print ("[STATUS] feature vector normalized...")

testDataGlobal=np.array(rescaled_features)
testLabelsGlobal=np.array(target)
"""
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                           np.array(target),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print( "[STATUS] splitted train and test data...")
print( "Train data  : {}".format(trainDataGlobal.shape))
print( "Test data   : {}".format(testDataGlobal.shape))
print( "Train labels: {}".format(trainLabelsGlobal.shape))
print( "Test labels : {}".format(testLabelsGlobal.shape))

# filter all the warnings
import warnings
import scipy
from scipy import stats
from scipy.stats import chi2
from scipy.stats import chi2_contingency
import pandas as pd 

warnings.filterwarnings('ignore')

train_labels = os.listdir(train_path)

# 10-fold cross validation
for name, model in models:
    
    #if(name == 'RF' or name =='LDA'):
        #continue

    model.fit(trainDataGlobal, trainLabelsGlobal)

    # without cross-valdidation
    prediction = model.predict(testDataGlobal)
    print(testLabelsGlobal)
    print(prediction)
    
    if(name != 'SVM'):
        prediction1 = model.predict_proba(testDataGlobal)

    predict1 = []
    original = []
    index = 0
    match = 0
    while (index < len(testLabelsGlobal) and index < len(prediction)):
        #print(train_labels[testLabelsGlobal[index]], train_labels[prediction[index]])
        if(testLabelsGlobal[index] == prediction[index]):
            match += 1
        index += 1
    
    print(name, "Accuracy: ", match/index)
    
    false_pos = 0
    true_neg = 0
    tot_pos = 0
    tot_neg = 0
    
    index = 0
    print(max(prediction))
    while (index < len(testLabelsGlobal) and index < len(prediction)):
        #print(train_labels[testLabelsGlobal[index]], train_labels[prediction[index]])
        
        if(testLabelsGlobal[index] != 0):
            if(testLabelsGlobal[index] != prediction[index]):
                false_pos += 1
            tot_pos += 1
            original.append(0)
            if(name != 'SVM'):
                predict1.append(1-prediction1[index][testLabelsGlobal[index]])
            else:
                if(testLabelsGlobal[index] != prediction[index]):
                    predict1.append(1)
                else:
                    predict1.append(0)
        elif(testLabelsGlobal[index] == 0):
            if(testLabelsGlobal[index] != prediction[index]):
                true_neg += 1
            tot_neg += 1
            original.append(1)
            if(name != 'SVM'):
                predict1.append(prediction1[index][testLabelsGlobal[index]])
            else:
                if(testLabelsGlobal[index] != prediction[index]):
                    predict1.append(0)
                else:
                    predict1.append(1)
        index += 1
    
    print(name, "False positive :", false_pos, tot_pos, false_pos/tot_pos)
    print(name, "True Negative :", true_neg, tot_neg, true_neg/tot_neg)
    
    if(1==1):
        from sklearn.metrics import roc_curve
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(original, predict1)
        
        
        from sklearn.metrics import auc
        auc_keras = auc(fpr_keras, tpr_keras)
        
        #print(prediction,testLabelsGlobal)
        
        exp_freq = []
        exp_freq.append(np.sum(original))
        exp_freq.append(len(original)-np.sum(original))
        
        obs_freq = []
        obs_freq.append(np.sum(predict1))
        obs_freq.append(len(predict1)-np.sum(predict1))
        
        
        #chi_sqr, chi1 = scipy.stats.chisquare(obs_freq, f_exp =exp_freq)
        
        #print(chi_sqr, chi1)
        """
        chi_sqr = 0
        for i in range(len(original)):
            chi_sqr += (original[i]- predict1[i])**2
            
        print(chi_sqr)
        p_val = 1-chi2.cdf(chi_sqr, 1)
        
        #p_val = chi1
        file1 = open(name + 'orig.txt', "w")
        for j in range(len(original)):
            file1.write(str(original[j]))
            file1.write('\n')
        file1.close()
        
        file1 = open(name + 'pred.txt', "w")
        for j in range(len(predict1)):
            file1.write(str(predict1[j]))
            file1.write('\n')
        file1.close()
        """
        
        
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label=name + ' ({:.3f}, p<0.0001)'.format(auc_keras),linewidth=2.0)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (Facial Dataset)')
        plt.legend(loc='best')

    """    
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    """

#Part 2 - fitting the data set
y_act = []
y_pred = []
import glob
Controlled = glob.glob('training_set/cats/*.*')
Diseased = glob.glob('training_set/dogs/*.*')

data = []
labels = []

for i in Controlled:   
    image=load_img(i, 
    target_size= (128,128))
    image=np.array(image)
    data.append(image)
    labels.append(0)
for i in Diseased:   
    image=load_img(i,  
    target_size= (128,128))
    image=np.array(image)
    data.append(image)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

from keras.utils.np_utils import to_categorical   

categorical_labels = to_categorical(labels, num_classes=2)


from sklearn.model_selection import train_test_split
X_train, X_test1, ytrain, ytest1 = train_test_split(data, categorical_labels, test_size=0.2,
                                                random_state=42)

X_val, X_test, yval, ytest = train_test_split(X_test1, ytest1, test_size=0.5,
                                                random_state=42)

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
        batch_size=64)

val_set = val_datagen.flow(
        X_val, yval,
        batch_size=64)

test_set = test_datagen.flow(
        X_test, ytest,
        batch_size=64)

X, y = test_set.next()

  
classifier.load_weights('DogCat_trained_model_weights.h5')

y_pred1 = classifier.predict(X);
y_pred2 = classifier.predict_classes(X);

for index in range(len(y_pred2)):
    y_act.append(np.argmax(y[index]))
    y_pred.append(np.argmax(y_pred1[index]))
    """
    if(np.argmax(y[index]) == 0):
        y_pred.append(1-y_pred1[index][0])
    else:
        y_pred.append(y_pred1[index][1])
   """
"""
file1 = open('cnn_orig.txt', "w")
for j in range(len(y_act)):
    file1.write(str(y_act[j]))
    file1.write('\n')
file1.close()

file1 = open('cnn_pred.txt', "w")
for j in range(len(y_pred1)):
    file1.write(str(np.argmax(y_pred1[j])))
    file1.write('\n')
file1.close()


chi_sqr = 0
for i in range(len(y_act)):
    chi_sqr += (y_act[i]- y_pred[i])**2
    
print(chi_sqr)
p_val = 1-chi2.cdf(chi_sqr, 1)
"""

from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_act, y_pred)

#print(cohen_kappa_score(y_act, y_pred))
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='CNN ({:.3f}, p<0.0001)'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (Dog_Cat Classification)')
plt.legend(loc='best')
plt.show()
"""
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100, random_state=9)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# path to test data
test_path = "dataset/test"

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    print(file)
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
"""