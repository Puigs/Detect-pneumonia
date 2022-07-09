#!/usr/bin/env python 

# Import for Geneterator image
import matplotlib.pyplot as plt #For Visualization
import numpy as np              #For handling arrays
import pandas as pd             # For handling data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import tensorflow as tf
# Import for IA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

import onnx
# import keras2onnx

import glob
# import cv2

#Define Directories for train, test & Validation Set
train_path = '../dataset/train'
test_path = '../dataset/test'
valid_path = '../dataset/val'
#Define some often used standard parameters
#The batch refers to the number of training examples utilized in one #iteration
batch_size = 16
#The dimension of the images we are going to define is 500x500 img_height = 500
img_width = 500
img_height = 500

# train = tf.keras.utils.image_dataset_from_directory(train_path)

# test = tf.keras.utils.image_dataset_from_directory(test_path)

# valid = tf.keras.utils.image_dataset_from_directory(valid_path)

#Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
                                  rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,          
                               )
# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale = 1./255)

train = image_gen.flow_from_directory(
      train_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary',
      batch_size=batch_size,
      )
test = test_data_gen.flow_from_directory(
      test_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      shuffle=False, 
#setting shuffle as False just so we can later compare it with predicted values without having indexing problem 
      class_mode='binary',
      batch_size=batch_size,
      )
valid = test_data_gen.flow_from_directory(
      valid_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary', 
      #batch_size=batch_size,
      )
# train.next()
# test.next()
# valid.next()

# train2 = test_data_gen.flow_from_directory(
#       "dataset/train/NORMAL",
#       target_size=(img_height, img_width),
#       color_mode='grayscale',
#       class_mode='binary',
#       batch_size=batch_size,
#       save_to_dir='dataset/resized/validation',
#       save_prefix='norm',
#       save_format='jpeg'
#       )
# train3 = test_data_gen.flow_from_directory(
#       "dataset/train",
#       target_size=(img_height, img_width),
#       color_mode='grayscale',
#       class_mode='binary',
#       #batch_size=batch_size,
#       save_to_dir='dataset/resized/validation',
#       save_prefix='norm',
#       save_format='jpeg'
#       )
# train2.next()
# train3.next()
# plt.figure(figsize=(12, 12))
# for i in range(0, 10):
#     plt.subplot(2, 5, i+1)
#     for X_batch, Y_batch in train:
#         image = X_batch[0]        
#         dic = {0:"NORMAL", 1:"PNEUMONIA"}
#         plt.title(dic.get(Y_batch[0]))
#         plt.axis("off")
#         plt.imshow(np.squeeze(image),cmap="gray",interpolation="nearest")
#         break
# plt.tight_layout()
#plt.show()
# def get_files(path, train):
#       files = glob.glob (path)
#       for myFile in files:
#             image = cv2.imread(myFile)
#             train.append (image)
#       return train

# train = get_files("dataset/train/NORMAL/*", [])
# train = get_files("dataset/train/PNEUMONIA/*", train)
# print(np.array(train)[0])
#exit()
cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'relu', units = 64))
cnn.add(Dense(activation = 'sigmoid', units = 1))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.summary()

# Hyperparameters of Conv2D
# Conv2D(filters,kernel_size,strides=(1, 1),padding="valid",activation=None,input_shape=(height,width,color, channel))
# # Hyperparameters of MaxPooling2D 
# MaxPooling2D(
#     pool_size=(2, 2), strides=None, padding="valid"
#     )

#plot_model(cnn,show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss", patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
callbacks_list = [ early, learning_rate_reduction]

from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight(class_weight = "balanced", classes = np.unique(train.classes), y = train.classes)
cw = dict(zip( np.unique(train.classes), weights))
print(cw)

cnn.fit(train,epochs=32, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

pd.DataFrame(cnn.history.history).plot()

test_accu = cnn.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')

new_test = cnn.evaluate(valid)
print('The testing accuracy is :',new_test[1]*100, '%')

preds = cnn.predict(test,verbose=1)

predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

from sklearn.metrics import classification_report,confusion_matrix
cm = pd.DataFrame(data=confusion_matrix(test.classes, predictions, labels=[0, 1]),index=["Actual Normal", "Actual Pneumonia"],
columns=["Predicted Normal", "Predicted Pneumonia"])
import seaborn as sns
sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_true=test.classes,y_pred=predictions,target_names =['NORMAL','PNEUMONIA']))

#test.reset()
#x=np.concatenate([test.next()[0] for i in range(test.__len__())])
#y=np.concatenate([test.next()[1] for i in range(test.__len__())])
#print(x.shape)
#print(y.shape)
#this little code above extracts the images from test Data iterator without shuffling the sequence
# x contains image array and y has labels 
#dic = {0:'NORMAL', 1:'PNEUMONIA'}
#plt.figure(figsize=(20,20))
#for i in range(0+228, 9+228):
#  plt.subplot(3, 3, (i-228)+1)
# if preds[i, 0] >= 0.5: 
 #     out = ('{:.2%} probability of being Pneumonia case'.format(preds[i][0]))
#  else: 
#      out = ('{:.2%} probability of being Normal case'.format(1-preds[i][0]))
#plt.title(out+"\n Actual case : "+ dic.get(y[i]))    
#plt.imshow(np.squeeze(x[i]))
#plt.axis('off')
#plt.show()

cnn.save('model-withoutReduce')