from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from Classification_model import Classification_model as Cm
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

labels = ["Ball", "Bottle", "Can", "Cup", "Face", "Pen", "Phone", "Shoe", "Silverware", "Yogurt"]

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # predict with the model
    start = timeit.timeit()
    preds = model.predict(image)
    end = timeit.timeit()
    print end - start

    #check if there is a object infront of the camera
    if np.argmax(preds) > no_object_threshold:
        print labels[np.argmax(preds, axis = 1).astype(np.int)]


