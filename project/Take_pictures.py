##Programm to take Pictures with the pi
##Save them in the folder
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)
i = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    if cv2.waitKey(1) & 0xFF == ord('1'):
        string = 'Pi_Pictures/feature1/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif cv2.waitKey(1) & 0xFF == ord('2'):
        string = 'Pi_Pictures/feature2/picture' + str(i) + '.png'
        cv2.imwrite(string, image)
        i += 1
    elif cv2.waitKey(1) & 0xFF == ord('3'):
        string = 'Pi_Pictures/feature3/picture' + str(i) + '.png'
        cv2.imwrite(string, image)
        i += 1
    elif cv2.waitKey(1) & 0xFF == ord('4'):
        string = 'Pi_Pictures/feature4/picture' + str(i) + '.png'
        cv2.imwrite(string, image)
        i += 1
    elif cv2.waitKey(1) & 0xFF == ord('5'):
        string = 'Pi_Pictures/feature5/picture' + str(i) + '.png'
        cv2.imwrite(string, image)
        i += 1
    
    cv2.imshow('frame', image)
    # the loop breaks at pressing `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
