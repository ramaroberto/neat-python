from __future__ import print_function, division

import math
import os
import sys

from fractions import Fraction

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
PRINT_LONG_NUMS_FOR_TESTING = False
PRINT_CHOOSING_AMONG = False

DO_ONLY = ("clamped_log1p_step")

CPPN_MAYBE_GROUP = ['hat_gauss_rectangular', 'multiparam_gauss',
                    'bicentral']

CPPN_DEFINITE_GROUP = ['fourth_square_abs',
                       'multiparam_log_inv',
                       'scaled_expanded_log',
                       'wave']
CPPN_NO_GROUP = CPPN_MAYBE_GROUP[:]

HIGH_NORM_EPSILON = round(math.ceil(NORM_EPSILON*(10**6))*(10**-6),6)

save_to_print = {}
save_exact_to_print = {}
save_to_print_abs = {}
strings_done = set([])

def print_for_testing(string, result, data):
    if not DO_PRINT_FOR_TESTING:
        return
    global strings_done, save_to_print, save_exact_to_print, save_to_print_abs
    if string in strings_done:
        return
    else:
        strings_done.add(string)

    try:
        ignored = len(data)
    except TypeError:
        print("TypeError from {0!r} ({1!r},{2!r})".format(data,string,result))
        raise
    global save_to_print, save_exact_to_print, save_to_print_abs
    result = float(result)
    name = "activations.{0}".format(string)
    if not (result or (math.copysign(1.0,result) > 0)):
        print("# Skipping {0} with result {1!r}".format(name,result)) # signed zero
        return
    rounded = round(result,6)
    if rounded == result:
        print("assert {0} == {1!r}".format(name, result))
    elif (abs(result-float("{0:.7g}".format(result)))
          < 1e-06) and (abs(result-round(result,0))
                        > NORM_EPSILON) and (abs(result-round(result,2))
                                             > math.sqrt(sys.float_info.epsilon)):
        save_result = round(result,min(8,sys.float_info.dig))
        if (abs(result-save_result) >= 1e-07):
            raise RuntimeError(
                "Result {0!r} vs save_result {1!r} (diff {2:n}, dig {3:n})".format(
                    result, save_result, abs(result-save_result), sys.float_info.dig))
        if save_result in save_to_print:
            save_to_print[save_result].append([name,result,data])
        else:
            save_to_print[save_result] = [[name,result,data]]
        if abs(save_result) in save_to_print_abs:
            save_to_print_abs[abs(save_result)].append([name,result,data])
        else:
            save_to_print_abs[abs(save_result)] = [[name,result,data]]
        if result in save_exact_to_print:
            save_exact_to_print[result].append([name,data])
        else:
            save_exact_to_print[result] = [[name,data]]
##        print("assert_almost_equal({0},{1!r})".format(name, result))
    elif (abs(result) >= sys.float_info.epsilon) and (abs(result) < NORM_EPSILON):
        if result > 0.0:
            print("assert 0.0 <= {0} <= {1!r}".format(name,HIGH_NORM_EPSILON))
        else:
            print("assert {0!r} <= {1} <= 0.0".format(-HIGH_NORM_EPSILON,name))
    elif ((abs(result)-1.0) >= sys.float_info.epsilon) and ((abs(result)-1.0) < NORM_EPSILON):
        if result > 0.0:
            print("assert 1.0 <= {0} <= {1!r}".format(name,round((1.0+HIGH_NORM_EPSILON),6)))
        else:
            print("assert {0!r} <= {1} <= -1.0".format(round((-1.0-HIGH_NORM_EPSILON),6)))
    else:
        print("# Skipping {0} with result {1!r}".format(name,result))

def get_log2_dist(a):
    a = abs(a)
    if a > 0.0:
        log2_a = abs(math.log(a,2))
        return abs(round(log2_a,0)-log2_a)
    return 0.0

def get_data_dists(data1, data2):
    try:
        len1 = len(data1)
        len2 = len(data2)
    except TypeError:
        print("Data1 {0!r} data2 {1!r}".format(data1,data2))
        raise
    
    if len1 != len2:
        raise ValueError(
            "Data1 {0!r} len {1:n} != data2 {2!r} len {3:n}".format(
                data1, len1, data2, len2))
    dist1 = 0.0
    dist2 = 0.0
    for a, b in zip(data1,data2):
        dist1 += abs(a-b)
        dist2 += get_log2_dist(a) + get_log2_dist(b)
    return [dist1,dist2]

def try_as_fraction(num):
    init_fraction = Fraction(num)
    ignored_integer, floating = math.modf(num)
    max_len = max(4,int(math.ceil((len(repr(abs(floating)))-2)/2.0)))
    round_fraction = init_fraction.limit_denominator(10**max_len)
    new_num = (round_fraction.numerator/round_fraction.denominator)
    if abs(new_num-num) < min(abs(round(num,7)-num),1e-07,math.sqrt(sys.float_info.epsilon)):
        return "({0!r}/{1!r})".format(round_fraction.numerator,
                                      round_fraction.denominator)
    return "{0!r}".format(num)

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
            poss_frac = try_as_fraction(save_to_print_abs[abs_result][0][1])
            if PRINT_LONG_NUMS_FOR_TESTING or ("/" in poss_frac):
                print("assert_almost_equal({0},{1})".format(save_to_print_abs[abs_result][0][0],
                                                            poss_frac))
                did_print_result.add(round(save_to_print_abs[abs_result][0][1],sys.float_info.dig))
                did_print_result_exact.add(save_to_print_abs[abs_result][0][1])
                did_print_result_abs.add(abs_result)
        elif len(save_to_print_abs[abs_result]) == 2:
            name1, result1, ignored_data1 = save_to_print_abs[abs_result][0]
            name2, result2, ignored_data2 = save_to_print_abs[abs_result][1]
            did_print_result.add(round(result1,sys.float_info.dig))
            did_print_result.add(round(result2,sys.float_info.dig))
            did_print_result_exact.add(result1)
            did_print_result_exact.add(result2)
            did_print_result_abs.add(abs_result)
            if abs(result1-result2) < 1e-06:
                print("assert_almost_equal({0},\n {1})".format(name1,name2))
            elif abs(result1+result2) < 1e-06:
                print("assert_almost_equal({0},\n -1*{1})".format(name1,name2))
            else:
                raise RuntimeError(
                    "{0} result abs({1!r}) != {2} result abs({3!r})".format(
                        name1, result1, name2, result2))
    save_to_print_abs = {}
    for save_result in sorted([n for n in iterkeys(save_to_print) if n not in did_print_result]):
        if len(save_to_print[save_result]) == 1:
            if abs(save_result) not in did_print_result_abs:
                poss_frac = try_as_fraction(save_to_print[save_result][0][1])
                if PRINT_LONG_NUMS_FOR_TESTING or ("/" in poss_frac):
                    print("assert_almost_equal({0},{1})".format(save_to_print[save_result][0][0],
                                                                poss_frac))
                    did_print_result_exact.add(save_to_print[save_result][0][1])
                    did_print_result_abs.add(abs(save_result))
        elif len(save_to_print[save_result]) == 2:
            name1, result1, ignored_data1 = save_to_print[save_result][0]
            name2, result2, ignored_data2 = save_to_print[save_result][1]
            if abs(result1-result2) < 1e-06:
                print("assert_almost_equal({0},\n {1})".format(name1,name2))
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
                poss_frac = try_as_fraction(result)
                if PRINT_LONG_NUMS_FOR_TESTING or ("/" in poss_frac):
                    print("assert_almost_equal({0},{1})".format(save_exact_to_print[result][0][0],
                                                                poss_frac))
                    did_print_result.add(rounded)
                    did_print_result_abs.add(abs_rounded)
        elif len(save_exact_to_print[result]) == 2:
            name1 = save_exact_to_print[result][0][0]
            name2 = save_exact_to_print[result][1][0]
            print("assert_almost_equal({0},\n {1})".format(name1,name2))
            did_print_result.add(rounded)
            did_print_result_abs.add(abs_rounded)
        else:
            if PRINT_CHOOSING_AMONG:
                print("#Choosing among {0:n} possibilities for result {1!r}".format(
                    len(save_exact_to_print[result]), result))
            dist1_dict = {}
            dist2_dict = {}
            poss = []
            start_num = 0
            for misc in save_exact_to_print[result]:
                poss.append([start_num,misc[0],misc[1]])
                start_num += 1
            for num1, name1, data1 in poss:
                for num2, name2, data2 in poss:
                    if num1 < num2:
                        dist1_dict[name1, name2], dist2_dict[name1, name2] = get_data_dists(data1,data2)
            poss_pairs = list(iterkeys(dist1_dict))
            poss_pairs.sort(key=lambda x: dist2_dict[x])
            poss_pairs.sort(reverse=True, key=lambda x: dist1_dict[x])
            name1 = poss_pairs[0][0]
            name2 = poss_pairs[0][1]
            print("assert_almost_equal({0},\n {1})".format(name1,name2))
            did_print_result.add(rounded)
            did_print_result_abs.add(abs_rounded)
    save_exact_to_print = {}

def trim_image(tmpname, realname):
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

num_subfigures = 5 # ODD NUMBERS ONLY!

x = np.linspace(-2.5, 2.5, 5000)

mps = MultiParameterSet('activation')
afs = ActivationFunctionSet(mps)
for n in sorted(iterkeys(mps.norm_func_dict['activation'])):
    if (DO_ONLY is not None) and (n not in DO_ONLY):
        continue
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
        print_for_testing("{0}_activation({1!s})".format(n,i),f(i),[i])
    do_prints()

def get_other_vars(mpf,name1,name2):
    if len(mpf.evolved_param_names) == 2:
        return {}
    to_return_dict = {}
    for name in [name3 for name3 in mpf.evolved_param_names if ((name3 != name1) and
                                                                (name3 != name2))]:
        dict3 = mpf.param_dicts[name].param_dict
        min_value_use = dict3.get('min_init_value', dict3['min_value'])
        max_value_use = dict3.get('max_init_value', dict3['max_value'])
        to_return_dict[name] = dict3.get('init_mean', ((min_value_use+max_value_use)/2.0))
    return to_return_dict

def format_dict(mpf,var_dict):
    to_join = []
    for n in mpf.evolved_param_names:
        if n in var_dict:
            to_join.append("{0}={1!r}".format(n, var_dict[n]))
        else:
            to_join.append(n)
    return ",".join(to_join)

for n in sorted(iterkeys(mps.multiparam_func_dict['activation'])):
    if (DO_ONLY is not None) and (n not in DO_ONLY):
        continue
    if (n not in CPPN_MAYBE_GROUP) and (n not in CPPN_DEFINITE_GROUP):
        CPPN_NO_GROUP.append(n)
    mpf = mps.multiparam_func_dict['activation'][n]
    f = mpf.user_func
    if len(mpf.evolved_param_names) == 1:
        name_nums = [tuple([0,0])]
    else:
        name_nums = []
        for i in range(len(mpf.evolved_param_names)-1):
            for j in range((i+1),len(mpf.evolved_param_names)):
                name_nums.append(tuple([i,j]))
                name_nums.append(tuple([j,i]))
    for name_num1, name_num2 in name_nums:
        param_name = mpf.evolved_param_names[name_num1]
        print("{0} dict for {1}: {2!r}".format(
            n, param_name, mpf.param_dicts[param_name].param_dict))
        if name_num1 != name_num2:
            param2_name = mpf.evolved_param_names[name_num2]
            print("{0} dict for {1}: {2!r}".format(
                n, param2_name, mpf.param_dicts[param2_name].param_dict))
        else:
            param2_name = None
        if param2_name is not None:
            fig = plt.figure(figsize=((5*num_subfigures),4))
        else:
            fig = plt.figure(figsize=((4*num_subfigures),4))
        plt.delaxes()
        other_vars = {}
        if param2_name is not None:
            other_vars = get_other_vars(mpf,param_name,param2_name)
        fig.suptitle("{0}(x,{1})".format(n, format_dict(mpf,other_vars)))
        dict1 = mpf.param_dicts[param_name].param_dict
        min_value = dict1.get('min_init_value', dict1['min_value'])
        max_value = dict1.get('max_init_value', dict1['max_value'])
        init_type = dict1.get('init_type', 'uniform')
        middle_param_value = dict1.get('init_mean', ((min_value+max_value)/2.0))
        if param2_name is not None:
            dict2 = mpf.param_dicts[param2_name].param_dict
            min_value2 = dict2.get('min_init_value', dict2['min_value'])
            max_value2 = dict2.get('max_init_value', dict2['max_value'])
            init_type2 = dict2.get('init_type', 'uniform')
        param_use = param_name
        param2_use = param2_name
        max_value_use = max_value
        min_value_use = min_value
        init_use = init_type
        if param2_name is not None:
            max_value2_use = max_value2
            min_value2_use = min_value2
            init2_use = init_type2
        param_value_list = [round(a,3) for a in list(np.linspace(max_value_use, min_value_use, num_subfigures))]
        if init_use.lower() in 'uniform':
            important_nums = (min_value_use,middle_param_value,max_value_use)
        elif init_use.lower() in ('gaussian', 'normal'):
            tmp_param_value_list = sorted(param_value_list, key=lambda tmp: abs(tmp-middle_param_value))
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
                param2_value_list = [round(b,3) for b in list(np.linspace(max_value2_use, min_value2_use, 5))]
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
                    other_vars.update({param_use:a, param2_use:b})
                    all_nums = [other_vars[name4] for name4 in mpf.evolved_param_names]
                    plt.plot(x, [f(i,**other_vars) for i in x], color, label="{0}={1}".format(param2_use,b))
                    if (color in important_colors) and (a in important_nums):
                        for i in (-1.0,-0.5,0.0,0.5,1.0):
                            print_for_testing("{0}_activation({1!s},{2!s})".format(n,i,format_dict(mpf,other_vars)),f(i,**other_vars),[i]+all_nums)
                    elif (color in important_colors) or (a in important_nums):
                        for i in (-1.0,0.0,1.0):
                            print_for_testing("{0}_activation({1!s},{2!s})".format(n,i,format_dict(mpf,other_vars)),f(i,**other_vars),[i]+all_nums)
            else:
                plt.plot(x, [f(i,a) for i in x])
                if a == middle_param_value:
                    for i in (-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0):
                        print_for_testing("{0}_activation({1!s},{2!r})".format(n,i,a),f(i,a),[i,a])                   
                else:
                    for i in (-1.0,-0.5,0.0,0.5,1.0):
                        print_for_testing("{0}_activation({1!s},{2!r})".format(n,i,a),f(i,a),[i,a])
            plt.title("{0}={1}".format(param_use, a))
            plt.grid()
            plt.xlim(-2.0, 2.0)
            plt.ylim(-2.0, 2.0)
            if param2_name is not None:
                plt.legend(loc='best')
                plt.gca().set_aspect(1)
        if len(mpf.evolved_param_names) > 2:
            tmpname = "activation-tmp-{0}-{1}-{2}.png".format(n,param_name,param2_name)
            realname = "activation-{0}-{1}-{2}.png".format(n,param_name,param2_name)
        elif name_num1 > name_num2:
            tmpname = "activation-tmp-swap-{0}.png".format(n)
            realname = "activation-swap-{0}.png".format(n)
        else:
            tmpname = "activation-tmp-{0}.png".format(n)
            realname = "activation-{0}.png".format(n)
        plt.savefig(tmpname)
        plt.close()
        trim_image(tmpname, realname)
    do_prints()

def do_funcs_for_name(funcs, name, group_name):
    y_size = min((5*num_subfigures),(4+math.floor(len(funcs)/3.0)))
    fig = plt.figure(figsize=((5*num_subfigures),y_size))
    plt.delaxes()
    fig.suptitle("{0} activation functions using {1}".format(group_name,name))
    dict1 = mps.shared_names['activation'][name].param_dict
    min_value = dict1.get('min_init_value', dict1['min_value'])
    max_value = dict1.get('max_init_value', dict1['max_value'])
    param_value_list = [round(a,3) for a in list(np.linspace(max_value, min_value, num_subfigures))]
    subplot_num = 0
    for a in param_value_list:
        subplot_num += 1
        fig.add_subplot(1,num_subfigures,subplot_num)

        for n in funcs:
            mpf = mps.multiparam_func_dict['activation'][n]
            all_vars = {name:a}
            for name2 in [name3 for name3 in mpf.evolved_param_names if (name3 != name)]:
                dict2 = mpf.param_dicts[name2].param_dict
                min_value_use = dict2.get('min_init_value', dict2['min_value'])
                max_value_use = dict2.get('max_init_value', dict2['max_value'])
                all_vars[name2] = dict2.get('init_mean', ((min_value+max_value)/2.0))
            f = mpf.user_func
            plt.plot(x, [f(i, **all_vars) for i in x], label=n)
        plt.title("{0}={1}".format(name, a))
        plt.grid()
        plt.xlim(-2.0, 2.0)
        plt.ylim(math.ceil(y_size*-0.5), 2.0)
        plt.legend(loc='best')
    tmpname = "shared-tmp-{0}-{1}.png".format(name,group_name)
    realname = "shared-{0}-{1}.png".format(name,group_name)
    plt.savefig(tmpname)
    plt.close()
    trim_image(tmpname, realname)

skipped_names = {}

for group in (CPPN_NO_GROUP, CPPN_MAYBE_GROUP, CPPN_DEFINITE_GROUP):
    shared_param_dict = {}
    if group == CPPN_NO_GROUP:
        group_name = 'Non-CPPN'
    elif group == CPPN_MAYBE_GROUP:
        group_name = 'Maybe-CPPN'
    else:
        group_name = 'CPPN'
    for n in group:
        mpf = mps.multiparam_func_dict['activation'][n]
        for name in mpf.evolved_param_names:
            if name in mps.shared_names['activation']:
                if name in shared_param_dict:
                    shared_param_dict[name].append(n)
                else:
                    shared_param_dict[name] = [n]
    for name in sorted(iterkeys(shared_param_dict)):
        funcs = shared_param_dict[name]
        if len(funcs) < 2:
            print("{0}: Only {1:n} func ({2!r}) for '{3}'".format(
                group_name,
                len(funcs),
                funcs,
                name))
            if name in skipped_names:
                skipped_names[name].add(funcs[0])
            else:
                skipped_names[name] = set(funcs)
            continue
        do_funcs_for_name(funcs, name, group_name)

for name in sorted(iterkeys(skipped_names)):
    funcs = skipped_names[name]
    if len(funcs) >= 2:
        do_funcs_for_name(funcs, name, 'Skipped')
    else:
        print("Only {0:n} func ({1!r}) for '{2}'".format(len(funcs),
                                                         funcs,
                                                         name))
