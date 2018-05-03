import cv2
import numpy as np
from Classification_model import Classification_model as Cm


# Script for training different models


cm1 = Cm("d(3_5)_s(64)_b(100)_e(400)_r(0_3)_w(0_1)_h(0_1)_z(0_3)_m1")
cm1.set_parameter(64,100,400,0.3,0.1,0.1,0.3,1)
cm1.load_model()
cm1.train_model()

cm2 = Cm("d(3_5)_s(64)_b(100)_e(400)_r(0_3)_w(0_1)_h(0_1)_z(0_1)_m1")
cm2.set_parameter(64,100,400,0.3,0.1,0.1,0.1,1)
cm2.load_model()
cm2.train_model()

cm3 = Cm("d(3_5)_s(64)_b(100)_e(500)_r(0_5)_w(0_2)_h(0_2)_z(0_5)_m1")
cm3.set_parameter(64,100,500,0.5,0.2,0.2,0.5,1)
cm3.load_model()
cm3.train_model()

cm4 = Cm("d(3_5)_s(64)_b(50)_e(600)_r(0_5)_w(0_3)_h(0_3)_z(0_3)_m1")
cm4.set_parameter(64,50,600,0.5,0.3,0.3,0.3,1)
cm4.load_model()
cm4.train_model()

cm5 = Cm("d(3_5)_s(64)_b(30)_e(800)_r(1)_w(0_5)_h(0_5)_z(0_9)_m1")
cm5.set_parameter(64,30,800,1,0.5,0.5,0.9,1)
cm5.load_model()
cm5.train_model()

cm6 = Cm("d(3_5)_s(64)_b(150)_e(400)_r(0_7)_w(0_1)_h(0_1)_z(0_6)_m1")
cm6.set_parameter(64,150,400,0.7,0.1,0.1,0.6,1)
cm6.load_model()
cm6.train_model()

cm7 = Cm("d(3_5)_x(64)_b(20)_e(400)_r(0_6)_w(0_3)_h(0_1)_z(0_5)_m1")
cm7.set_parameter(64,20,400,0.6,0.3,0.1,0.5,1)
cm7.load_model()
cm7.train_model()

cm8 = Cm("d(3_5)_s(64)_b(100)_e(400)_r(0_3)_w(0_1)_h(0_1)_z(0_3)_m2")
cm8.set_parameter(64,100,400,0.3,0.1,0.1,0.3,2)
cm8.load_model()
cm8.train_model()

cm9 = Cm("d(3_5)_s(64)_b(100)_e(400)_r(0_3)_w(0_1)_h(0_1)_z(0_1)_m2")
cm9.set_parameter(64,100,400,0.3,0.1,0.1,0.1,2)
cm9.load_model()
cm9.train_model()

cm10 = Cm("d(3_5)_s(64)_b(100)_e(500)_r(0_5)_w(0_2)_h(0_2)_z(0_5)_m2")
cm10.set_parameter(64,100,500,0.5,0.2,0.2,0.5,2)
cm10.load_model()
cm10.train_model()

cm11 = Cm("d(3_5)_s(64)_b(50)_e(600)_r(0_5)_w(0_3)_h(0_3)_z(0_3)_m2")
cm11.set_parameter(64,50,600,0.5,0.3,0.3,0.3,2)
cm11.load_model()
cm11.train_model()

cm12 = Cm("d(3_5)_s(32)_b(30)_e(800)_r(1)_w(0_5)_h(0_5)_z(0_9)_m2")
cm12.set_parameter(64,30,800,1,0.5,0.5,0.9,2)
cm12.load_model()
cm12.train_model()

cm13 = Cm("d(3_5)_s(32)_b(150)_e(400)_r(0_7)_w(0_1)_h(0_1)_z(0_6)_m2")
cm13.set_parameter(64,150,400,0.7,0.1,0.1,0.6,2)
cm13.load_model()
cm13.train_model()

cm14 = Cm("d(3_5)_x(32)_b(20)_e(400)_r(0_6)_w(0_3)_h(0_1)_z(0_5)_m2")
cm14.set_parameter(64,20,400,0.6,0.3,0.1,0.5,2)
cm14.load_model()
cm14.train_model()
