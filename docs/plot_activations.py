from __future__ import print_function

import math
import os

from sys import float_info

import matplotlib.pyplot as plt
import numpy as np

try:
    from Pillow import Image, ImageChops
except ImportError:
    from PIL import Image, ImageChops

from neat.multiparameter import MultiParameterSet
from neat.activations import ActivationFunctionSet
from neat.math_util import median2
from neat.six_util import iterkeys

def print_for_testing(string, result):
    result = float(result)
    if result or (math.copysign(1.0,result) > 0):
        name = "activations.{0}".format(string)
    else:
        name = "abs(activations.{0})".format(string) # signed 0 problem
        result = abs(result)
    rounded = round(result,6)
    if rounded == result:
        print("assert {0} == {1!r}".format(name, result))
    elif (abs(result-float("{0:.7g}".format(result)))
          < 1e-06) and (abs(result-round(result,3)) > math.sqrt(float_info.epsilon)):
        print("assert_almost_equal({0},{1!r})".format(name, result))
    else:
        print("# Skipping {0} with result {1!r}".format(name,result))

num_subfigures = 5

x = np.linspace(-2.5, 2.5, 5000)

mps = MultiParameterSet('activation')
afs = ActivationFunctionSet(mps)
for n in sorted(iterkeys(mps.norm_func_dict['activation'])):
    f = mps.norm_func_dict['activation'][n]
    plt.figure(figsize=(4, 4))
    plt.plot(x, [f(i) for i in x])
    plt.title(n)
    plt.grid()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect(1)
    plt.savefig('activation-{0}.png'.format(n))
    plt.close()
    for i in (-1.0,-0.5,0.0,0.5,1.0):
        print_for_testing("{0}_activation({1!s})".format(n,i),f(i))

for n in sorted(iterkeys(mps.multiparam_func_dict['activation'])):
    mpf = mps.multiparam_func_dict['activation'][n]
    f = mpf.user_func
    if len(mpf.evolved_param_names) > 2: # NOTE: EVENTUALLY ALSO NEED TO CHECK FOR NON-FLOAT!
        print("Cannot currently handle 3+ evolved parameters (function {!s})".format(n))
        continue
    elif len(mpf.evolved_param_names) > 1:
        param2_name = mpf.evolved_param_names[1]
        swap=[False,True]
    else:
        param2_name = None
        swap=[False]
    for do_swap in swap:
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
        if do_swap:
            param_use = param2_name
            param2_use = param_name
            max_value_use = max_value2
            min_value_use = min_value2
            max_value2_use = max_value
            min_value2_use = min_value
        else:
            param_use = param_name
            param2_use = param2_name
            max_value_use = max_value
            min_value_use = min_value
            if param2_name is not None:
                max_value2_use = max_value2
                min_value2_use = min_value2
        
        param_value_list = np.linspace(max_value_use, min_value_use, num_subfigures)
        middle_param_value = median2(param_value_list)
        subplot_num = 0
        for a in param_value_list:
            subplot_num += 1
            fig.add_subplot(1,num_subfigures,subplot_num)
            if param2_name is not None:
                param2_value_list = np.linspace(max_value2_use, min_value2_use, 5)
                for b, color in zip(param2_value_list, ['c-','g--','b-','r--','m-']):
                    if do_swap:
                        plt.plot(x, [f(i,b,a) for i in x], color, label="{0}={1}".format(param2_use,b))
                    else:
                        plt.plot(x, [f(i,a,b) for i in x], color, label="{0}={1}".format(param2_use,b))
                        if (color in ('c-', 'b-', 'm-')) and (a in (min_value_use,middle_param_value,max_value_use)):
                            for i in (-1.0,0.0,1.0):
                                print_for_testing("{0}_activation({1!s},{2!r},{3!r})".format(n,i,a,b),f(i,a,b))
            else:
                plt.plot(x, [f(i,a) for i in x])
                if a == middle_param_value:
                    for i in (-1.0,-0.5,0.0,0.5,1.0):
                        print_for_testing("{0}_activation({1!s},{2!r})".format(n,i,a),f(i,a))                   
                else:
                    for i in (-1.0,0.0,1.0):
                        print_for_testing("{0}_activation({1!s},{2!r})".format(n,i,a),f(i,a))
            plt.title("{0}={1}".format(param_use, a))
            plt.grid()
            plt.xlim(-2.0, 2.0)
            plt.ylim(-2.0, 2.0)
            if param2_name is not None:
                plt.legend()
                plt.gca().set_aspect(1)
        if do_swap:
            tmpname = "activation-tmp-swap-{0}.png".format(n)
            realname = "activation-swap-{0}.png".format(n)
        else:
            tmpname = "activation-tmp-{0}.png".format(n)
            realname = "activation-{0}.png".format(n)
        plt.savefig(tmpname)
        plt.close()
        img = Image.open(tmpname)
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            new_img = img.crop(bbox)
            new_img.save(realname)
        else:
            img.save(realname)
        os.unlink(tmpname)
