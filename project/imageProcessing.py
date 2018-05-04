import cv2
import numpy as np

def resize_to_item(image, batch_shape):

    size = (batch_shape[0],batch_shape[1])

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 1801, 41)
    _, contours, _ = cv2.findContours(thresh, 1, 2)
    idx = -1  # The index of the contour that surrounds your object
    mask = np.zeros_like(image)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, idx, 255, 1)  # Draw filled contour in mask

    # Now crop
    (y, x) = np.where(mask == 255)
    (topx, topy) = (np.max(x), np.max(y))
    (bottomx, bottomy) = (np.min(x), np.min(y))

    rightBorder = False
    leftBorder = False
    topBorder = False
    bottomBorder = False

    if(topx == 640): rightBorder = True
    if(bottomx == 0):leftBorder = True
    if (topy == 480): bottomtBorder = True
    if (bottomy == 0): topBorder = True

    topx+=40
    topy+=40
    bottomx-=40
    bottomy-=40

    if(topx >= 640): topx = 640
    if(topy >= 480): topy = 480
    if(bottomx <= 0): bottomx = 0
    if(bottomy <= 0): bottomy = 0

    height = topy - bottomy
    width = topx - bottomx
    centerx = int(bottomx+width/2)
    centery = int(bottomy + height/2)

    if(topBorder & bottomBorder):
        topx = int(centerx + height/2)
        bottomx = int(centerx - height/2)
    elif(rightBorder):
        topx = bottomx + height
        if(topx >= 640):
            topx = 640
            bottomx = topx - height
    elif(leftBorder):
        bottomx = topx - height
        if(bottomx <= 0):
            bottomx = 0
            topx = bottomx + height
    elif(topBorder):
        bottomy = topy - width
        if(bottomy <= 0):
            bottomy = 0
            topy = bottomy + width
    elif(bottomBorder):
        topy = bottomy + width
        if(topy >= 480):
            topy = 480
            bottomy = topy-width

    batch = image[bottomy:topy, bottomx:topx]

    batch = cv2.resize(batch, size)

    return batch