import numpy as np
import cv2
import matplotlib.pyplot as plt
from Dataset_generator import *

(a,b,c,d) = load_train_set()
cv2.imshow("image", a[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
