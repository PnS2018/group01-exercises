from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from keras.models import Model
import numpy as np

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

model = create_model()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # predict with the model
    preds = np.argmax(model.predict(test_x_vis), axis=1).astype(np.int)
