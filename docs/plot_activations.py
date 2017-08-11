from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from Pillow import Image, ImageChops
except ImportError:
    from PIL import Image, ImageChops

from neat.multiparameter import MultiParameterSet
from neat.activations import ActivationFunctionSet

num_subfigures = 5

x = np.linspace(-2.5, 2.5, 5000)

mps = MultiParameterSet('activation')
afs = ActivationFunctionSet(mps)
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
    if len(mpf.evolved_param_names) > 2:
        print("Cannot currently handle 3+ evolved parameters (function {!s})".format(n))
        continue
    elif len(mpf.evolved_param_names) > 1:
        param2_name = mpf.evolved_param_names[1]
    else:
        param2_name = None
    param_name = mpf.evolved_param_names[0]
    if param2_name is not None:
        fig = plt.figure(figsize=((5*num_subfigures),4))
    else:
        fig = plt.figure(figsize=((4*num_subfigures),4))
    plt.delaxes()
    if param2_name is not None:
        fig.suptitle("{0}(x,{1},{2})".format(n,param_name,param2_name))
    else:
        fig.suptitle("{0}(x,{1})".format(n, param_name))
    min_value = mpf.evolved_param_dicts[param_name]['min_value']
    max_value = mpf.evolved_param_dicts[param_name]['max_value']
    if param2_name is not None:
        min_value2 = mpf.evolved_param_dicts[param2_name]['min_value']
        max_value2 = mpf.evolved_param_dicts[param2_name]['max_value']
    param_value_list = np.linspace(max_value, min_value, num_subfigures)
    subplot_num = 0
    for a in param_value_list:
        subplot_num += 1
        fig.add_subplot(1,num_subfigures,subplot_num)
        if param2_name is not None:
            param2_value_list = np.linspace(max_value2, min_value2, 5)
            for b, color in zip(param2_value_list, ['c-','g--','b-','r--','m-']):
                plt.plot(x, [f(i,a,b) for i in x], color, label="{0}={1}".format(param2_name,b))
        else:
            plt.plot(x, [f(i,a) for i in x])
        plt.title("{0}={1}".format(param_name, a))
        plt.grid()
        plt.xlim(-2.0, 2.0)
        plt.ylim(-2.0, 2.0)
        if param2_name is not None:
            plt.legend()
        plt.gca().set_aspect(1)
    plt.savefig('activation-tmp-{0}.png'.format(n))
    plt.close()
    img = Image.open("activation-tmp-{0}.png".format(n))
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        new_img = img.crop(bbox)
        new_img.save("activation-{0}.png".format(n))
    else:
        img.save("activation-{0}.png".format(n))
    os.unlink("activation-tmp-{0}.png".format(n))
