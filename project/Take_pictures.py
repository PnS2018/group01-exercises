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
f = open('Pi_Pictures/last_pic_index.txt', "r")
i = int(f.read())
f.close()
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    key = cv2.waitKey(100)
    if key==ord('1'):
        string = 'Pi_Pictures/Balls/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('2'):
        string = 'Pi_Pictures/Bottles/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('3'):
        string = 'Pi_Pictures/Cans/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('4'):
        string = 'Pi_Pictures/Cups/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('5'):
        string = 'Pi_Pictures/Face/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('6'):
        string = 'Pi_Pictures/Pens/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('7'):
        string = 'Pi_Pictures/Phone/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('8'):
        string = 'Pi_Pictures/Shoes/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('9'):
        string = 'Pi_Pictures/Silverware/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    elif key==ord('0'):
        string = 'Pi_Pictures/Yoghurt/picture' + str(i) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i += 1
    # the loop breaks at pressing `q`
    elif key==ord('q'):
        string = 'Pi_Pictures/last_pic_index.txt'
        f = open(string, "w")
        f.write(str(i))
        f.close()
        break
    cv2.imshow('frame', image)

    

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
