"""Convolutional Neural Network for Fashion MNIST Classification.

Team #name
"""
from __future__ import print_function, absolute_import

import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing import image


from pnslib import utils
from pnslib import ml

# Load all the ten classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels
(train_x, train_y, test_x, test_y) = utils.fashion_mnist_load(
    data_type="full", flatten=False)

num_classes = 10

print ("[MESSAGE] Dataset is loaded.")

# preprocessing for training and testing images
train_x = train_x.astype("float32")/255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32")/255.
test_x -= mean_train_x

print ("[MESSAGE] Dataset is preprocessed.")

# converting the input class labels to categorical labels for training
train_Y = to_categorical(train_y, num_classes=num_classes)
test_Y = to_categorical(test_y, num_classes=num_classes)

print("[MESSAGE] Converted labels to categorical labels.")

# define a model
# >>>>> PUT YOUR CODE HERE <<<<<

input_shape = train_x.shape[1:]

x = Input(input_shape)
y = Conv2D(filters =20,
           kernel_size=(7,7),
           padding="same",
           activation="relu",
           )(x)
y = MaxPooling2D((2,2), strides=(2,2))(y)
y = Conv2D(filters =25,
           kernel_size=(5,5),
           padding="same",
           activation="relu",
           )(y)
y = MaxPooling2D((2,2), strides=(2,2))(y)
y = Flatten()(y)
y = Dense(10, activation="softmax", )(y)
model = Model(x,y)


print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

# compile the model aganist the categorical cross entropy loss
# and use SGD optimizer, you can try to use different
# optimizers if you want
# see https://keras.io/losses/
# >>>>> PUT YOUR CODE HERE <<<<<
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["mse", "accuracy"])



print ("[MESSAGE] Model is compiled.")

# train the model with fit function
# See https://keras.io/models/model/ for usage
# >>>>> PUT YOUR CODE HERE <<<<<


datagen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_x)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_x, train_Y, batch_size=32),
                    steps_per_epoch=len(train_x) / 32, epochs=1)


print("[MESSAGE] Model is trained.")

# save the trained model
model.save("conv-net-fashion-mnist-trained.hdf5")

print("[MESSAGE] Model is saved.")

# visualize the ground truth and prediction
# take first 10 examples in the testing dataset
test_x_vis = test_x[:10]  # fetch first 10 samples
ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
# predict with the model
preds = np.argmax(model.predict(test_x_vis), axis=1).astype(np.int)

labels = ["Tshirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle Boot"]

plt.figure()
for i in xrange(2):
    for j in xrange(5):
        plt.subplot(2, 5, i*5+j+1)
        plt.imshow(np.squeeze(test_x[i*5+j]), cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[ground_truths[i*5+j]],
                   labels[preds[i*5+j]]))
plt.show()
