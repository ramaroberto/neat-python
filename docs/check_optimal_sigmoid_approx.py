from __future__ import print_function, division

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from Pillow import Image, ImageChops
except ImportError:
    from PIL import Image, ImageChops

from neat.multiparameter import MultiParameterSet
from neat.activations import ActivationFunctionSet
from neat.math_util import median2, NORM_EPSILON
from neat.six_util import iterkeys

def tanh_approx(z, a, b, tanh_mult=1.0):
    if z >= (sys.float_info.max/b):
        return 1.0
    if z <= (-1*sys.float_info.max/b):
        return -1.0
    to_return = (b*z)/(math.pow(b,a) + abs(b*z))
    return min(1.0,max(-1.0,(to_return*tanh_mult)))

def sigmoid_approx(z, a, b,tanh_mult=1.0,sigmoid_mult=2.0):
    return min(1.0,max(0.0,((1.0+tanh_approx((sigmoid_mult*z),a,b,tanh_mult))/2.0)))

def chebyshev_points(N):
    assert isinstance(N,int)
    if N == 1:
        return [0.5]
    return [math.cos(x*math.pi/(N-1)) for x in range(N)]

def get_linspace_list(start,finish,digits):
    num_numbers = int(round(abs(finish-start)*(10**digits)))
    to_return = []
    for n in np.linspace(start,finish,num_numbers):
        to_return.append(round(n,digits))
    return to_return

#x_list = np.linspace(-1.0, 1.0, 1000)
x_list = chebyshev_points(100)
a_list = get_linspace_list(-0.2399,-0.23988,6)
b_list = get_linspace_list(5.88778,5.887885,6)
sm_list = get_linspace_list(1.10548,1.105474,7)

mps = MultiParameterSet('activation')
afs = ActivationFunctionSet(mps)

sigmoid_func = mps.norm_func_dict['activation']['sigmoid']
tanh_func = mps.norm_func_dict['activation']['tanh']

best_diff = sys.float_info.max
best_a_was = None
best_b_was = None
best_mult = 1.0
best_sm = 2.0

sigmoid_result = {}
tanh_result = {}

for x in x_list:
    sigmoid_result[x] = sigmoid_func(x)
    tanh_result[x] = tanh_func(x)

tanh_at_5 = tanh_func(5.0)

list_done = set()
for sm in sm_list:
    if best_a_was is not None:
        print("Doing sm={0!r}; best_diff {1!r}, a {2:n}, b {3!r}, sm {4!r}".format(sm,best_diff,best_a_was,best_b_was,best_sm))
    for b in b_list:
        for a in a_list:
            what = tuple([a,b,sm])
            if what in list_done:
                continue
            list_done.add(what)

            tanh_approx_mult = 1.0
            if tanh_approx(5.0,a,b) < tanh_at_5:
                tanh_approx_mult = tanh_at_5/tanh_approx(5.0,a,b)

            total_diff = math.fsum([abs(tanh_result[x]-tanh_approx(x,a,b,tanh_approx_mult)) for x in x_list])
            total_diff += 0.25*math.fsum([abs(sigmoid_result[x]-sigmoid_approx(x,a,b,tanh_approx_mult,sigmoid_mult=sm)) for x in x_list])
            if total_diff < best_diff:
                best_diff = total_diff
                best_a_was = a
                best_b_was = b
                best_mult = tanh_approx_mult
                best_sm = sm

print("Best a was {0!r}, best b was {1!r} (best_diff/{2:n} {3!r}; mult {4!r}, sm {5!r})".format(
    best_a_was,best_b_was,len(x_list),(best_diff/len(x_list)), best_mult, best_sm))

x = np.linspace(-2.5, 2.5, 5000)

plt.figure(figsize=(4, 4))
plt.plot(x, [sigmoid_func(i) for i in x], 'g-', label='sigmoid')
plt.plot(x, [sigmoid_approx(i,best_a_was,best_b_was,best_mult,best_sm) for i in x], 'r-', label='sigmoid_approx')
plt.grid()
plt.title('sigmoid vs sigmoid_approx(a={0:n},b={1:n})'.format(best_a_was,best_b_was))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect(1)
plt.savefig('sigmoid_comparison.png')
plt.close()

plt.figure(figsize=(4, 4))
plt.plot(x, [tanh_func(i) for i in x], 'g-', label='tanh')
plt.plot(x, [(best_mult*tanh_approx(i,best_a_was,best_b_was)) for i in x], 'r-', label='tanh_approx')
plt.grid()
plt.title('tanh vs tanh_approx(a={0:n},b={1:n})'.format(best_a_was,best_b_was))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect(1)
plt.savefig('tanh_comparison.png')
plt.close()
