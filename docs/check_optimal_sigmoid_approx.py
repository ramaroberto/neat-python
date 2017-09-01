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



num_subfigures = 5 # ODD NUMBERS ONLY!

x_list = np.linspace(-1.0, 1.0, 5000)
a_list = [(round((2.0*a),2)/2.0) for a in np.linspace(-1.6, 0.0, 500)]

mps = MultiParameterSet('activation')
afs = ActivationFunctionSet(mps)

sigmoid_func = mps.norm_func_dict['activation']['sigmoid']
mpf = mps.multiparam_func_dict['activation']['multiparam_sigmoid_approx']
sigmoid_approx_func = mpf.user_func

a_list_done = set()

best_diff = sys.float_info.max
best_was = None

for a in a_list:
    if a in a_list_done:
        continue
    a_list_done.add(a)

    total_diff = sum([abs(sigmoid_func(x)-sigmoid_approx_func(x,a)) for x in x_list])
    if total_diff < best_diff:
        best_diff = total_diff
        best_was = a

print("Best a was {0:n} (best_diff/5000 {1:n})".format(best_was,(best_diff/5000)))

x = np.linspace(-2.5, 2.5, 5000)

plt.figure(figsize=(4, 4))
plt.plot(x, [sigmoid_func(i) for i in x], 'g-', label='sigmoid')
plt.plot(x, [sigmoid_approx_func(i,best_was) for i in x], 'r-', label='sigmoid_approx')
plt.grid()
plt.title('sigmoid vs sigmoid_approx(a={0:n})'.format(best_was))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect(1)
plt.savefig('sigmoid_comparison.png')
plt.close()
