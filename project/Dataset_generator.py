"""Function to define our training Dataset"""

"""output: (train_x,train_y,Feature_number, shape) """
"""shape = (width, height,numberchannel)"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
Feature_number = 10
PicturesPFeature = 1
color = 0
shape = (640,480,1) #1 because greyscale


def load_train_set():
    train_x = np.zeros((Feature_number*PicturesPFeature, 480, 640), dtype=np.uint8)
    train_y = np.zeros((Feature_number*PicturesPFeature, 1))
    for i in range (0,Feature_number):
        for k in range (0,PicturesPFeature):
            if i == 0:
                feature = 'Pi_Pictures/Train/Balls/0_picture'
            elif i == 1:
                feature = 'Pi_Pictures/Train/Bottles/1_picture'
            elif i == 2:
                feature = 'Pi_Pictures/Train/Cans/2_picture'
            elif i == 3:
                feature = 'Pi_Pictures/Train/Cups/3_picture'
            elif i == 4:
                feature = 'Pi_Pictures/Train/Face/4_picture'
            elif i == 5:
                feature = 'Pi_Pictures/Train/Pens/5_picture'
            elif i == 6:
                feature = 'Pi_Pictures/Train/Phone/6_picture'
            elif i == 7:
                feature = 'Pi_Pictures/Train/Shoes/7_picture'
            elif i == 8:
                feature = 'Pi_Pictures/Train/Silverware/8_picture'
            elif i == 9:
                feature = 'Pi_Pictures/Train/Yoghurt/9_picture'
            train_y[i*PicturesPFeature+k] = int(i)
            string = feature + str(k) + '.png'
            train_x[i*PicturesPFeature+k] = cv2.imread(string, color)

    output = (train_x, train_y, Feature_number, shape)
    return output

def load_valid_set():

    valid_x = []
    valid_y = []

    output = (valid_x, valid_y)
    return output

