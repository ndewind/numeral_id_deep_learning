# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:03:06 2019

Adapting the data camp tutorial code to do numeral identification

data camp tutorial for Keras:
https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

@author: Nick
"""
import numpy as np
import matplotlib.pyplot as plt

# import the dataset from keras
from keras.datasets import mnist
(train_X,train_Y), (test_X,test_Y) = mnist.load_data()

# display the shape of the data
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# check the number and data type of the classes
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
print('Output class type : ', classes.dtype)

# plot
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

# reshape the data
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, test_X.shape

# retype the data
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
import keras.utils as utils
train_Y_one_hot = utils.to_categorical(train_Y)
test_Y_one_hot = utils.to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# split training and testing sets
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


# ...setup the deep CNN!... #

# import keras 
import keras
from keras.models import Sequential,Input,Model,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# define the model with dropout layers.
batch_size = 64
epochs = 20
num_classes = 10
numeral_model = Sequential()
numeral_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
numeral_model.add(LeakyReLU(alpha=0.1))
numeral_model.add(MaxPooling2D((2, 2),padding='same'))
numeral_model.add(Dropout(0.25))
numeral_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
numeral_model.add(LeakyReLU(alpha=0.1))
numeral_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
numeral_model.add(Dropout(0.25))
numeral_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
numeral_model.add(LeakyReLU(alpha=0.1))                  
numeral_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
numeral_model.add(Dropout(0.4))
numeral_model.add(Flatten())
numeral_model.add(Dense(128, activation='linear'))
numeral_model.add(LeakyReLU(alpha=0.1))           
numeral_model.add(Dropout(0.3))
numeral_model.add(Dense(num_classes, activation='softmax'))
numeral_model.summary()

# compile the dropout model
numeral_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

# fit the model
numeral_train_dropout = numeral_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# save the model (so we don't have to train it again)
numeral_model.save("numeral_model_dropout.h5py")

# load the model (if you want to avoid retraining it)
numeral_model = load_model("numeral_model_dropout.h5py")

# evaluate the model
test_eval = numeral_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# and make some plots
accuracy = numeral_train_dropout.history['acc']
val_accuracy = numeral_train_dropout.history['val_acc']
loss = numeral_train_dropout.history['loss']
val_loss = numeral_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# predictions!
predictions = numeral_model.predict(test_X, verbose=1)
predictionsInt = np.argmax(predictions, axis=1)

# plot some predictions
plt.figure(figsize=[20,30])
for x in range(25):
    # Display the first image in testing data
    plt.subplot(5,5,x+1)
    plt.imshow(test_X[x,:,:,0], cmap='gray')
    titleStr = "Ground Truth : {}\nEstimate : {}".format(test_Y[x],predictionsInt[x])
    plt.title(titleStr)
    
# plot some errors
errLogIndx = predictionsInt != test_Y
errWhereIndx = np.where(errLogIndx)[0]
plt.figure(figsize=[20,30])
for x in range(25):
    # Display the first image in testing data
    plt.subplot(5,5,x+1)
    plt.imshow(test_X[errWhereIndx[x],:,:,0], cmap='gray')
    titleStr = "Ground Truth : {}\nEstimate : {}".format(test_Y[errWhereIndx[x]],predictionsInt[errWhereIndx[x]])
    plt.title(titleStr)

    