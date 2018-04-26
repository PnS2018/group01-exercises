from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from Classification_model import Classification_model as Cm
from Dataset_generator import *
import timeit

#define no object threshold
no_object_threshold = 0.8

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

cm = Cm("conv_net")
cm.load_model()
#cm.train_model()
model = cm.get_model()



num_class = cm.get_number_of_classes()
(resize_x, resize_y) = get_resize()


labels = ["Ball", "Bottle", "Can", "Cup", "Face", "Pen", "Phone", "Shoe", "Silverware", "Yogurt"]

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    image_resized = cv2.resize(image, (0,0), fx=resize_x, fy=resize_y)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) 
    # predict with the model
    start = timeit.timeit()
    preds = model.predict(np.expand_dims(np.expand_dims(image_gray, axis=0),axis =3))
    end = timeit.timeit()
    print end - start

    #check if there is a object infront of the camera
    if np.argmax(preds) > no_object_threshold:
        print labels[np.argmax(preds).astype(np.int)]

    rawCapture.truncate(0)


