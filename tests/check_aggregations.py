from __future__ import print_function, division

import math
import os
import sys
import warnings

from operator import itemgetter

import numpy as np

from neat.multiparameter import MultiParameterSet
from neat.aggregations import AggregationFunctionSet
from neat.math_util import NORM_EPSILON, median2
from neat.six_util import iterkeys

warnings.simplefilter('default')

save_to_print = {}
save_exact_to_print = {}
save_to_print_abs = {}

PRINT_LONG_NUMS_FOR_TESTING = False

def print_for_testing(string, result):
    global save_to_print, save_exact_to_print, save_to_print_abs
    result = float(result)
    if result or (math.copysign(1.0,result) > 0):
        name = "aggregations.{0}".format(string)
    else:
        name = "abs(aggregations.{0})".format(string) # signed 0 problem
        result = abs(result)
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
    global save_to_print, save_exact_to_print, save_to_print_abs
    if not len(save_to_print):
        return
    did_print_result_abs = set([])
    did_print_result = set([])
    did_print_result_exact = set([])
    for abs_result in sorted(iterkeys(save_to_print_abs)):
        if len(save_to_print_abs[abs_result]) == 1:
            if PRINT_LONG_NUMS_FOR_TESTING:
                print("assert_almost_equal({0},{1!r})".format(*save_to_print_abs[abs_result][0]))
                did_print_result.add(round(save_to_print_abs[abs_result][0][1],min(8,sys.float_info.dig)))
                did_print_result_exact.add(save_to_print_abs[abs_result][0][1])
                did_print_result_abs.add(abs_result)
        elif len(save_to_print_abs[abs_result]) == 2:
            name1, result1 = save_to_print_abs[abs_result][0]
            name2, result2 = save_to_print_abs[abs_result][1]
            did_print_result.add(round(result1,min(8,sys.float_info.dig)))
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
            if PRINT_LONG_NUMS_FOR_TESTING and (abs(save_result) not in did_print_result_abs):
                print("assert_almost_equal({0},{1!r})".format(*save_to_print_abs[abs_result][0]))
                did_print_result_exact.add(save_to_print[save_result][0][1])
                did_print_result_abs.add(abs(save_result))
        elif len(save_to_print[save_result]) == 2:
            name1, result1 = save_to_print[save_result][0]
            name2, result2 = save_to_print[save_result][1]
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
        rounded = round(result,min(8,sys.float_info.dig))
        abs_rounded = round(abs(result),min(8,sys.float_info.dig))
        if len(save_exact_to_print[result]) == 1:
            if PRINT_LONG_NUMS_FOR_TESTING and (rounded not in did_print_result) and (abs_rounded not in did_print_result_abs):
                print("assert_almost_equal({0},{1!r})".format(save_exact_to_print[result][0],result))
                did_print_result.add(rounded)
                did_print_result_abs.add(abs_rounded)
        elif len(save_exact_to_print[result]) == 2:
            name1 = save_exact_to_print[result][0]
            name2 = save_exact_to_print[result][1]
            print("assert_almost_equal({0},\n {1})".format(name1,name2))
            did_print_result.add(rounded)
            did_print_result_abs.add(abs_rounded)
        else:
            print("Not sure which to use for result {0!r}:\n\t".format(result)
                  + "\n\t".join(save_exact_to_print[result]))
    save_exact_to_print = {}
        
                

to_check_set = set([])
for a in [-1.0,0.0,1.0]:
    for b in [-0.5,0.5]:
        for c in [-1.0,0.0,1.0]:
            for d in [-0.5,0.5]:
                for e in [-1.0,0.0,1.0]:
                    to_check_set.add(tuple(sorted([a,b,c,d,e])))
print("Have {0:n} members of to_check_set".format(len(to_check_set)))
to_check_list = sorted(list(to_check_set), key=itemgetter(0,1,2,3,4))

to_check_set2 = set([])
for a in [-1.0,-0.5,0.0,0.5,1.0]:
    for b in [-0.5,0.5]:
        for c in [-2.0,-1.0,0.0,1.0,2.0]:
            for d in [-0.5,0.5]:
                to_check_set2.add(tuple(sorted([a,b,c,d])))
print("Have {0:n} members of to_check_set2".format(len(to_check_set2)))
to_check_list2 = sorted(list(to_check_set2), key=itemgetter(0,1,2,3))

to_check_set3 = set([]) # subset of to_check_set2
for a in [-1.0,0.0,1.0]:
    for b in [-0.5,0.5]:
        for c in [-2.0,-1.0,0.0,1.0,2.0]:
            for d in [-0.5,0.5]:
                to_check_set3.add(tuple(sorted([a,b,c,d])))
print("Have {0:n} members of to_check_set3".format(len(to_check_set3)))
to_check_list3 = sorted(list(to_check_set3), key=itemgetter(0,1,2,3))

to_check_set4 = set([])
for a in [-1.0,0.0,1.0]:
    for b in [-0.5,0.0,0.5]:
        for c in [-1.0,-0.5,0.0,0.5,1.0]:
            for d in [-0.5,0.0,0.5]:
                if len(set([a,b,c,d])) == 4:
                    combo = sorted([a,b,c,d])
                    for i in [0,1,2]:
                        num1 = (combo[i]+(3.0*combo[i+1]))/4.0
                        num2 = ((3.0*combo[i])+combo[i+1])/4.0
                        poss1 = tuple(sorted([a,b,c,d,num1]))
                        poss2 = tuple(sorted([a,b,c,d,num2]))
                        if poss1 not in to_check_set:
                            to_check_set4.add(poss1)
                        if poss2 not in to_check_set:
                            to_check_set4.add(poss2)
print("Have {0:n} members of to_check_set4".format(len(to_check_set4)))
to_check_list4 = sorted(list(to_check_set4), key=itemgetter(0,1,2,3,4))

to_check_list14 = to_check_list4 + to_check_list

mps = MultiParameterSet('aggregation')
afs = AggregationFunctionSet(mps)
for n in sorted(iterkeys(mps.norm_func_dict['aggregation'])):
    f = mps.norm_func_dict['aggregation'][n]
    result1_set = set([])
    for i in to_check_list:
        func_result = f(i)
        if (func_result not in result1_set) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
            print_for_testing("{0}_aggregation({1!r})".format(n,list(i)),func_result)
            result1_set.add(func_result)
    result2_set = set([])
    result3_set = set([])
    for i in to_check_list3:
        func_result = f(i)
        if (func_result not in result3_set) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
            print_for_testing("{0}_aggregation({1!r})".format(n,list(i)),func_result)
            if func_result in result2_set:
                result3_set.add(func_result)
            else:
                result2_set.add(func_result)
    for i in to_check_list2:
        func_result = f(i)
        if (func_result not in result2_set) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
            print_for_testing("{0}_aggregation({1!r})".format(n,list(i)),func_result)
            result2_set.add(func_result)
    do_prints()

for n in sorted(iterkeys(mps.multiparam_func_dict['aggregation'])):
    mpf = mps.multiparam_func_dict['aggregation'][n]
    f = mpf.user_func
    if len(mpf.evolved_param_names) > 3:
        print("Cannot currently handle 4+ evolved parameters (function {!s})".format(n))
        continue
    elif len(mpf.evolved_param_names) > 1:
        param2_name = mpf.evolved_param_names[1]
        param_dict2 = mpf.param_dicts[param2_name].param_dict
        num_diff = 3
        if param_dict2['param_type'] not in ('bool','float'):
            print("Cannot currently handle non-float/bool evolved parameters (function {!s}, type {!r})".format(
                n,param_dict2['param_type']))
            continue
        if len(mpf.evolved_param_names) > 2:
            param3_name = mpf.evolved_param_names[2]
            param_dict3 = mpf.param_dicts[param3_name].param_dict
            if param_dict3['param_type'] not in ('bool','float'):
                print("Cannot currently handle non-float/bool evolved parameters (function {!s}, type {!r})".format(
                    n,param_dict3['param_type']))
                continue
        else:
            param3_name = None
    else:
        param2_name = None
        param3_name = None
        num_diff = 5
    param_name = mpf.evolved_param_names[0]
    param_dict1 = mpf.param_dicts[param_name].param_dict
    min_value = param_dict1.get('min_init_value', param_dict1['min_value'])
    max_value = param_dict1.get('max_init_value', param_dict1['max_value'])
    middle_param_value = param_dict1.get('init_mean', ((min_value+max_value)/2.0))
    if (param2_name is not None) and (param_dict2['param_type'] == 'float'):
        min_value2 = param_dict2.get('min_init_value', param_dict2['min_value'])
        max_value2 = param_dict2.get('max_init_value', param_dict2['max_value'])
        middle_param2_value = param_dict2.get('init_mean', ((min_value2+max_value2)/2.0))
    if (param3_name is not None) and (param_dict3['param_type'] == 'float'):
        min_value3 = param_dict3.get('min_init_value', param_dict3['min_value'])
        max_value3 = param_dict3.get('max_init_value', param_dict3['max_value'])
        middle_param3_value = param_dict3.get('init_mean', ((min_value3+max_value3)/2.0))
    param_value_list = np.linspace(max_value, min_value, num_diff)
    if len(param_value_list) > 3:
        if n == 'max_median_min':
            param_value_list = [param_value_list[1], param_value_list[3], middle_param_value, min_value, max_value]
        else:
            param_value_list = [middle_param_value, min_value, max_value, param_value_list[1], param_value_list[3]]
    else:
        param_value_list = [middle_param_value, min_value, max_value]
    result_set = set([])
    if param2_name is not None:
        if param_dict2['param_type'] == 'float':
            param2_value_list = [middle_param2_value, min_value2, max_value2]
        else:
            param2_value_list = [False,True]
        if param3_name is not None:
            if param_dict3['param_type'] == 'float':
                param3_value_list = [middle_param3_value, min_value3, max_value3]
            else:
                param3_value_list = [False,True]
    elif n == 'max_median_min':
        param_value_list_full = np.linspace(max_value, min_value, 9)
        param_value_list_full = sorted(param_value_list_full, key=lambda x: abs(x-middle_param_value))
        for a in param_value_list_full:
            if (a == middle_param_value) or (a == max_value) or (a == min_value):
                continue
            equiv_percentile = 100*((a+1.0)/2.0)
            for i in to_check_list4:
                numpy_result = np.percentile(i,equiv_percentile)
                func_result = f(i,a)
                if abs(numpy_result-func_result) > 1e-06:
                    print("max_median_min({0!r}, {1!r}) is {2!r}; np.percentile({0!r}, {3!r}) is {4!r}".format(
                        list(i), a, func_result, equiv_percentile, numpy_result))
                elif func_result or (math.copysign(1.0,func_result) > 0.0):
                    print_for_testing("{0}_aggregation({1!r}, {2!r})".format(n,list(i),a),func_result)
                    result_set.add(func_result)

    if param2_name is None:
        for i in to_check_list14:
            for a in param_value_list:
                func_result = f(i,a)
                if (func_result not in result_set) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
                    print_for_testing("{0}_aggregation({1!r}, {2!r})".format(n,list(i),a),func_result)
                    result_set.add(func_result)
    for i in to_check_list3:
        for a in param_value_list:
            if param2_name is not None:
                for b in param2_value_list:
                    if param3_name is not None:
                        for c in param3_value_list:
                            func_result = f(i,a,b,c)
                            if (func_result not in result_set) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
                                print_for_testing("{0}_aggregation({1!r}, {2!r}, {3!r}, {4!r})".format(n,list(i),a,b,c),func_result)
                                result_set.add(func_result)
                    else:
                        func_result = f(i,a,b)
                        if ((((a == middle_param_value) and
                              ((param_dict2['param_type'] == 'bool') or (b == middle_param2_value)))
                             or (func_result != round(func_result,6))
                             or (func_result not in result_set)) and (func_result or (math.copysign(1.0,func_result) > 0.0))):
                            print_for_testing("{0}_aggregation({1!r}, {2!r}, {3!r})".format(n,list(i),a,b),func_result)
                            if a != middle_param_value:
                                result_set.add(func_result)

            else:
                func_result = f(i,a)
                if ((func_result != round(func_result,6))
                    or (func_result not in result_set)) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
                    print_for_testing("{0}_aggregation({1!r}, {2!r})".format(n,list(i),a),func_result)
                    if a != middle_param_value:
                        result_set.add(func_result)


                
                

    do_prints()

