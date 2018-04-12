from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.utils import to_categorical

from Dataset_generator import *

class Classification_model:
    def __init__(self,designator):
        self.designator = designator
        self.x = Input(shape)

        self.y = Conv2D(filters=20,
                   kernel_size=(7, 7),
                   padding="same",
                   activation="relu",
                   )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                   kernel_size=(5, 5),
                   padding="same",
                   activation="relu",
                   )(self.y)
        self.y = self.MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation="softmax", )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")


    def train_model(self):
        # load training dataset
        # shape = (width,height,channels) i.e shape = (224,256,3) for a 224x256 (widthxheight) with 3 RGB channels
        (train_x, train_y, self.num_classes, self.shape) = load_train_set()
        (valid_x, valid_y) = load_valid_set()
        print("[MESSAGE] Loaded testing dataset.")

        # converting the input class labels to categorical labels for training
        train_Y = to_categorical(train_y, num_classes=self.num_classes)
        valid_Y = to_categorical(valid_y, num_classes=self.num_classes)
        print("[MESSAGE] Converted labels to categorical labels.")

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
        batch_size = 20
        epochs = 5
        self.model.fit_generator(datagen.flow(train_x, train_Y, batch_size=batch_size),
                            steps_per_epoch=len(train_x) / batch_size, epochs=epochs)

        print("[MESSAGE] Model is trained.")

        # save the trained model
        self.model.save(self.designator + ".hdf5")
        print("[MESSAGE] Model is saved.")

    def load_model(self):
        try:
            my_abs_path = file.resolve()
        except FileNotFoundError:
            #doesn't exist
            print("[MESSAGE] No safed model weights found")
        else:
            # exists
            self.model.load_weights(self.designator + ".hdf5")
            print("[MESSAGE] Loaded model weights")