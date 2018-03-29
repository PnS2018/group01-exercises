from __future__ import print_function, absolute_import

import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 3
img1_color = cv2.imread('Lenna.png')
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)

# open the camera
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img2_color = frame
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)


    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2_color = cv2.polylines(img2_color, [np.int32(dst)],
                                   True, [0, 0, 255], 10, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(
            len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    img3 = cv2.drawMatches(img1_color, kp1, img2_color, kp2, good, None,
                           **draw_params)
    cv2.imshow('frame', img3)

    # the loop breaks at pressing `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
