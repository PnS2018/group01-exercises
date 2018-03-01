#Session 1 Exercise Answer
#Group 03 / Dominic Streiff and Jonas Binz

import numpy as np
from keras import backend as K

#Ex 1
a = K.placeholder(shape=(5,))
b = K.placeholder(shape=(5,))
c = K.placeholder(shape=(5,))

output = a**2 + b**2 + c**2 + 2*b*c

cos_function = K.function(inputs=(a,b,c),outputs=(output,))

a_1 = np.arange(5)
b_1 = np.arange(5)
c_1 = np.arange(5)
print('Ex 1')
print(cos_function((a_1,b_1,c_1)))

#Ex 2
x = K.ones(shape=())
tanh = (K.exp(x)-K.exp(-x))/(K.exp(x)+K.exp(-x))
grad_tanh = K.gradients(loss=tanh, variables=[x])

tanh_all = K.function(inputs=(x,), outputs=(tanh,grad_tanh[0]))

print('Ex 2')
print(tanh_all((10,)))

#Ex 3
w = K.ones(shape=(2,))
b = K.ones(shape=(1,))

x = K.placeholder(shape=(2,))

fun = 1/(1+K.exp(x[0]*w[0]+x[1]*w[1]+b))
grad_fun = K.gradients(loss=fun, variables=[w])

fun_all = K.function(inputs=(w,b,x), outputs=[fun] + grad_fun)

print('Ex 3')
print(fun_all((np.array([1,1]),np.array([3]),np.array([1,1]))))

#Ex 4
n = 4

variable = K.ones(shape=(n+1,))
x = K.placeholder(shape=())

poly = None
i=0
for element in variable:
    if poly is None:
        poly = element*x**i
    else:
        poly+=element*x**i
    i+=1

#grad_poly = K.gradients(loss=poly, variables
#poly_fun = K.function(inputs=(x,variable), outputs=[poly] + grad_poly)





                        
