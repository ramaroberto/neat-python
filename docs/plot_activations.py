from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from neat.activations import ActivationFunctionSet

num_subfigures = 5

x = np.linspace(-5.0, 5.0, 5000)

afs = ActivationFunctionSet()
mps = afs.multiparameterset
for n, f in mps.norm_func_dict['activation'].items():
    plt.figure(figsize=(4, 4))
    plt.plot(x, [f(i) for i in x])
    plt.title(n)
    plt.grid()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect(1)
    plt.savefig('activation-{0}.png'.format(n))
    plt.close()
for n, mpf in mps.multiparam_func_dict['activation'].items():
    f = mpf.user_func
    if (len(mpf.evolved_param_dicts) > 1):
        print("Cannot currently handle 2+ evolved parameters (function {!s})".format(n))
        next
    param_name = mpf.evolved_param_names[0]
    fig = plt.figure(figsize=((4*num_subfigures),4))
    plt.delaxes()
    fig.suptitle("{0}(x,{1})".format(n, param_name))
    min_value = mpf.evolved_param_dicts[param_name]['min_value']
    max_value = mpf.evolved_param_dicts[param_name]['max_value']
    param_value_list = np.linspace(max_value, min_value, num_subfigures)
    subplot_num = 0
    for a in param_value_list:
        subplot_num += 1
        fig.add_subplot(1,num_subfigures,subplot_num)
        plt.plot(x, [f(i,a) for i in x])
        plt.title("{0}={1}".format(param_name, a))
        plt.grid()
        plt.xlim(-2.0, 2.0)
        plt.ylim(-2.0, 2.0)
        plt.gca().set_aspect(1)
    plt.savefig('activation-{0}.png'.format(n))
    plt.close()
