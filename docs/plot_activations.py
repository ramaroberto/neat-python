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

DO_PRINT_FOR_TESTING = True

save_to_print = {}
save_exact_to_print = {}
save_to_print_abs = {}

def print_for_testing(string, result):
    if not DO_PRINT_FOR_TESTING:
        return
    global save_to_print, save_exact_to_print, save_to_print_abs
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
          < 1e-06) and (abs(result-round(result,0))
                        > NORM_EPSILON) and (abs(result-round(result,2))
                                             > math.sqrt(sys.float_info.epsilon)):
        save_result = round(result,sys.float_info.dig)
        if (abs(result-save_result) >= 1e-07):
            raise RuntimeError(
                "Result {0!r} vs save_result {1!r} (diff {2:n}, dig {3:n})".format(
                    result, save_result, abs(result-save_result), sys.float_info.dig))
        if save_result in save_to_print:
            save_to_print[save_result].append([name,result])
        else:
            save_to_print[save_result] = [[name,result]]
        if abs(save_result) in save_to_print_abs:
            save_to_print_abs[abs(save_result)].append([name,result])
        else:
            save_to_print_abs[abs(save_result)] = [[name,result]]
        if result in save_exact_to_print:
            save_exact_to_print[result].append(name)
        else:
            save_exact_to_print[result] = [name]
##        print("assert_almost_equal({0},{1!r})".format(name, result))
    else:
        print("# Skipping {0} with result {1!r}".format(name,result))

def do_prints():
    if not DO_PRINT_FOR_TESTING:
        return
    global save_to_print, save_exact_to_print, save_to_print_abs
    if not len(save_to_print):
        return
    did_print_result_abs = set([])
    did_print_result = set([])
    did_print_result_exact = set([])
    for abs_result in sorted(iterkeys(save_to_print_abs)):
        if len(save_to_print_abs[abs_result]) == 1:
            print("assert_almost_equal({0},{1!r})".format(*save_to_print_abs[abs_result][0]))
            did_print_result.add(round(save_to_print_abs[abs_result][0][1],sys.float_info.dig))
            did_print_result_exact.add(save_to_print_abs[abs_result][0][1])
            did_print_result_abs.add(abs_result)
        elif len(save_to_print_abs[abs_result]) == 2:
            name1, result1 = save_to_print_abs[abs_result][0]
            name2, result2 = save_to_print_abs[abs_result][1]
            did_print_result.add(round(result1,sys.float_info.dig))
            did_print_result.add(round(result2,sys.float_info.dig))
            did_print_result_exact.add(result1)
            did_print_result_exact.add(result2)
            did_print_result_abs.add(abs_result)
            if abs(result1-result2) < 1e-06:
                print("assert_almost_equal({0},{1})".format(name1,name2))
            elif abs(result1+result2) < 1e-06:
                print("assert_almost_equal({0},-1*{1})".format(name1,name2))
            else:
                raise RuntimeError(
                    "{0} result abs({1!r}) != {2} result abs({3!r})".format(
                        name1, result1, name2, result2))
    save_to_print_abs = {}
    for save_result in sorted([n for n in iterkeys(save_to_print) if n not in did_print_result]):
        if len(save_to_print[save_result]) == 1:
            if abs(save_result) not in did_print_result_abs:
                print("assert_almost_equal({0},{1!r})".format(*save_to_print[save_result][0]))
                did_print_result_exact.add(save_to_print[save_result][0][1])
                did_print_result_abs.add(abs(save_result))
        elif len(save_to_print[save_result]) == 2:
            name1, result1 = save_to_print[save_result][0]
            name2, result2 = save_to_print[save_result][1]
            if abs(result1-result2) < 1e-06:
                print("assert_almost_equal({0},{1})".format(name1,name2))
            else:
                raise RuntimeError(
                    "{0} result {1!r} != {2} result {3!r}".format(
                        name1, result1, name2, result2))
            did_print_result_exact.add(result1)
            did_print_result_exact.add(result2)
            did_print_result_abs.add(abs(save_result))
    save_to_print = {}
    for result in sorted([n for n in iterkeys(save_exact_to_print) if n not in did_print_result_exact]):
        rounded = round(result,sys.float_info.dig)
        abs_rounded = round(abs(result),sys.float_info.dig)
        if len(save_exact_to_print[result]) == 1:
            if (rounded not in did_print_result) and (abs_rounded not in did_print_result_abs):
                print("assert_almost_equal({0},{1!r})".format(save_exact_to_print[result][0],result))
                did_print_result.add(rounded)
                did_print_result_abs.add(abs_rounded)
        elif len(save_exact_to_print[result]) == 2:
            name1 = save_exact_to_print[result][0]
            name2 = save_exact_to_print[result][1]
            print("assert_almost_equal({0},{1})".format(name1,name2))
            did_print_result.add(rounded)
            did_print_result_abs.add(abs_rounded)
        else:
            print("Not sure which to use for result {0!r}:\n\t".format(result)
                  + "\n\t".join(save_exact_to_print[result]))
    save_exact_to_print = {}

num_subfigures = 5 # ODD NUMBERS ONLY!

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
    for i in (-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0):
        print_for_testing("{0}_activation({1!s})".format(n,i),f(i))
    do_prints()

for n in sorted(iterkeys(mps.multiparam_func_dict['activation'])):
    mpf = mps.multiparam_func_dict['activation'][n]
    f = mpf.user_func
    if len(mpf.evolved_param_names) > 2: # NOTE: EVENTUALLY ALSO NEED TO CHECK FOR NON-FLOAT!
        print("Cannot currently handle 3+ evolved parameters (function {0!s}: {1!r})".format(n,f))
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
        dict1 = mpf.evolved_param_dicts[param_name]
        min_value = dict1.get('min_init_value', dict1['min_value'])
        max_value = dict1.get('max_init_value', dict1['max_value'])
        init_type = dict1.get('init_type', 'uniform')
        if param2_name is not None:
            dict2 = mpf.evolved_param_dicts[param2_name]
            min_value2 = dict2.get('min_init_value', dict2['min_value'])
            max_value2 = dict2.get('max_init_value', dict2['max_value'])
            init_type2 = dict2.get('init_type', 'uniform')
        if do_swap:
            param_use = param2_name
            param2_use = param_name
            max_value_use = max_value2
            min_value_use = min_value2
            max_value2_use = max_value
            min_value2_use = min_value
            init_use = init_type2
            init2_use = init_type
        else:
            param_use = param_name
            param2_use = param2_name
            max_value_use = max_value
            min_value_use = min_value
            init_use = init_type
            if param2_name is not None:
                max_value2_use = max_value2
                min_value2_use = min_value2
                init2_use = init_type2

        if init_use.lower() in 'uniform':
            param_value_list = np.linspace(max_value_use, min_value_use, num_subfigures)
            middle_param_value = median2(param_value_list)
            important_nums = (min_value_use,middle_param_value,max_value_use)
        elif init_use.lower() in ('gaussian', 'normal'):
            param_value_list = [round(a,2) for a in np.linspace(max_value_use, min_value_use, (num_subfigures+2))]
            del param_value_list[-2]
            del param_value_list[1]
            middle_param_value = median2(param_value_list)
            tmp_param_value_list = sorted(param_value_list, key=lambda x: abs(x-middle_param_value))
            important_nums = (middle_param_value, tmp_param_value_list[1], tmp_param_value_list[2])
        else:
            raise ValueError(
                "{0}: Unknown init_type {1!r} for param_use '{2}'".format(
                    n, init_use, param_use))
        subplot_num = 0
        for a in param_value_list:
            subplot_num += 1
            fig.add_subplot(1,num_subfigures,subplot_num)
            if param2_name is not None:
                param2_value_list = np.linspace(max_value2_use, min_value2_use, 5)
                if init2_use.lower() in ('gaussian','normal'):
                    colors_use = ['c--','g-','b-','r-','m--']
                    important_colors = ('g-','b-','r-')
                elif init2_use.lower() in 'uniform':
                    colors_use = ['c-','g--','b-','r--','m-']
                    important_colors = ('c-', 'b-', 'm-')
                else:
                    raise ValueError(
                        "{0}: Unknown init_type {1!r} for param2_use '{2}'".format(
                            n, init2_use, param2_use))
                for b, color in zip(param2_value_list, colors_use):
                    if do_swap:
                        plt.plot(x, [f(i,b,a) for i in x], color, label="{0}={1}".format(param2_use,b))
                    else:
                        plt.plot(x, [f(i,a,b) for i in x], color, label="{0}={1}".format(param2_use,b))
                        if (color in important_colors) and (a in important_nums):
                            for i in (-1.0,-0.5,0.0,0.5,1.0):
                                print_for_testing("{0}_activation({1!s},{2!r},{3!r})".format(n,i,a,b),f(i,a,b))
                        elif (color in important_colors) or (a in important_nums):
                            for i in (-1.0,0.0,1.0):
                                print_for_testing("{0}_activation({1!s},{2!r},{3!r})".format(n,i,a,b),f(i,a,b))
            else:
                plt.plot(x, [f(i,a) for i in x])
                if a == middle_param_value:
                    for i in (-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0):
                        print_for_testing("{0}_activation({1!s},{2!r})".format(n,i,a),f(i,a))                   
                else:
                    for i in (-1.0,-0.5,0.0,0.5,1.0):
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
        if not do_swap:
            do_prints()
