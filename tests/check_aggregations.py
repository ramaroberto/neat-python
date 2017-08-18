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
                print("assert_almost_equal({0},{1!r})".format(*save_to_print_abs[abs_result][0]))
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

to_check_set3 = set([])
for a in [-1.0,0.0,1.0]:
    for b in [-0.5,0.5]:
        for c in [-2.0,-1.0,0.0,1.0,2.0]:
            for d in [-0.5,0.5]:
                to_check_set3.add(tuple(sorted([a,b,c,d])))
print("Have {0:n} members of to_check_set3".format(len(to_check_set3)))
to_check_list3 = sorted(list(to_check_set3), key=itemgetter(0,1,2,3))

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
    if len(mpf.evolved_param_names) > 2:
        print("Cannot currently handle 3+ evolved parameters (function {!s})".format(n))
        continue
    elif len(mpf.evolved_param_names) > 1:
        param2_name = mpf.evolved_param_names[1]
        num_diff = 3
        if mpf.evolved_param_dicts[param2_name]['param_type'] not in ('bool','float'):
            print("Cannot currently handle non-float/bool evolved parameters (function {!s})".format(n))
            continue
    else:
        param2_name = None
        num_diff = 5
    param_name = mpf.evolved_param_names[0]
    min_value = mpf.evolved_param_dicts[param_name]['min_value']
    max_value = mpf.evolved_param_dicts[param_name]['max_value']
    if (param2_name is not None) and (mpf.evolved_param_dicts[param2_name]['param_type'] == 'float'):
        min_value2 = mpf.evolved_param_dicts[param2_name]['min_value']
        max_value2 = mpf.evolved_param_dicts[param2_name]['max_value']
    param_value_list = np.linspace(max_value, min_value, num_diff)
    middle_param_value = median2(param_value_list)
    if len(param_value_list) > 3:
        param_value_list = [middle_param_value, min_value, max_value, param_value_list[1], param_value_list[3]]
    else:
        param_value_list = [middle_param_value, min_value, max_value]
    result_set = set([])
    if (param2_name is not None) and (mpf.evolved_param_dicts[param2_name]['param_type'] == 'float'):
        param2_value_list = np.linspace(max_value2, min_value2, 3)
        middle_param2_value = param2_value_list[1]
        param2_value_list = [middle_param2_value, min_value2, max_value2]
    for i in to_check_list3:
        for a in param_value_list:
            if param2_name is not None:
                if mpf.evolved_param_dicts[param2_name]['param_type'] == 'float':
                    for b in param2_value_list:
                        func_result = f(i,a,b)
                        if (((a == middle_param_value) and (b == middle_param2_value)) or (func_result != round(func_result,6))
                            or (func_result not in result_set)) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
                            print_for_testing("{0}_aggregation({1!r}, {2!r}, {3!r})".format(n,list(i),a,b),func_result)
                            result_set.add(func_result)
            
                else:
                    for b in (False,True):
                        func_result = f(i,a,b)
                        if ((func_result != round(func_result,6))
                            or (func_result not in result_set)) and (func_result or (math.copysign(1.0,func_result) > 0.0)):
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

