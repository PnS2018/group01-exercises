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
i = [0]*10
lines = f.readlines()
for k in range (0, 10):
    i[k] = int(lines[k])
f.close()
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    key = cv2.waitKey(100)
    if key==ord('1'):
        string = 'Pi_Pictures/Balls/0_picture' + str(i[0]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[0] += 1
    elif key==ord('2'):
        string = 'Pi_Pictures/Bottles/1_picture' + str(i[1]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[1] += 1
    elif key==ord('3'):
        string = 'Pi_Pictures/Cans/2_picture' + str(i[2]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[2] += 1
    elif key==ord('4'):
        string = 'Pi_Pictures/Cups/3_picture' + str(i[3]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[3] += 1
    elif key==ord('5'):
        string = 'Pi_Pictures/Face/4_picture' + str(i[4]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[4] += 1
    elif key==ord('6'):
        string = 'Pi_Pictures/Pens/5_picture' + str(i[5]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[5] += 1
    elif key==ord('7'):
        string = 'Pi_Pictures/Phone/6_picture' + str(i[6]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[6] += 1
    elif key==ord('8'):
        string = 'Pi_Pictures/Shoes/7_picture' + str(i[7]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[7] += 1
    elif key==ord('9'):
        string = 'Pi_Pictures/Silverware/8_picture' + str(i[8]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[8] += 1
    elif key==ord('0'):
        string = 'Pi_Pictures/Yoghurt/9_picture' + str(i[9]) + '.png'
        print(string)
        cv2.imwrite(string, image)
        i[9] += 1
    # the loop breaks at pressing `q`
    elif key==ord('q'):
        string = 'Pi_Pictures/last_pic_index.txt'
        f = open(string, "w")
        for l in range(0,10):
            f.write(str(i[l]))
            f.write("\n")
        f.close()
        break
    cv2.imshow('frame', image)

    

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
