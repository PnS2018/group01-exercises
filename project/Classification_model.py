from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model

from keras.utils import to_categorical

def save_trained_model(location):
    #load training dataset
    #shape = (width,height,channels) i.e shape = (224,256,3) for a 224x256 (widthxheight) with 3 RGB channels
    (train_x, train_y, num_classes, shape) = Dataset_generator.load_train_set()
    (valid_x,valid_y) = Dataset_generator.load_valid_set()
    print("[MESSAGE] Loaded testing dataset.")

    # converting the input class labels to categorical labels for training
    train_Y = to_categorical(train_y, num_classes=num_classes)
    valid_Y = to_categorical(valid_y, num_classes=num_classes)
    print("[MESSAGE] Converted labels to categorical labels.")

    x = Input(shape)

    y = Conv2D(filters=20,
               kernel_size=(7, 7),
               padding="same",
               activation="relu",
               )(x)
    y = MaxPooling2D((2, 2), strides=(2, 2))(y)
    y = Conv2D(filters=25,
               kernel_size=(5, 5),
               padding="same",
               activation="relu",
               )(y)
    y = MaxPooling2D((2, 2), strides=(2, 2))(y)
    y = Flatten()(y)
    y = Dense(num_classes, activation="softmax", )(y)
    model = Model(x, y)

    print("[MESSAGE] Model is defined.")

    # print model summary
    model.summary()


    # compile the model aganist the categorical cross entropy loss and use SGD optimizer
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["mse", "accuracy"])
    print ("[MESSAGE] Model is compiled.")

    # train the model with fit function
    model.fit(
        x=train_x, y=train_Y,
        batch_size=64, epochs=3,
        validation_data=(valid_x, valid_Y))

    print("[MESSAGE] Model is trained.")

    # save the trained model
    model.save("conv-net-fashion-mnist-trained.hdf5")
    print("[MESSAGE] Model is saved.")
