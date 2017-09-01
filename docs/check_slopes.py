from __future__ import print_function, division

import itertools
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

from neat.multiparameter import MultiParameterSet
from neat.activations import ActivationFunctionSet
from neat.math_util import median2, NORM_EPSILON
from neat.six_util import iterkeys

FULL_TEST = True

x = np.linspace(-2, 2, 1000)

def do_theilsen(y_data, x_data):
    slope, ignored_intercept, low, high = stats.theilslopes(y_data, x_data, alpha=0.25)
    intercept_list = []
    for x_use, y_use in zip(x_data,y_data):
        intercept_list.append(y_use - (slope*x_use)) # could check low, high admittedly...
    intercept = median2(intercept_list)
    return slope, intercept, low, high

def get_slope_intercept(n, f, checking, checking_range, names, all_dict, other_params, checking_val=None, done_params=None, param_values_list=None):
    if done_params is None:
        done_params = set([])
    if param_values_list is None:
        param_values_list = []
    to_return_dict = {}
    need = [i for i in names if i not in done_params]
    if need:
        new_done_params = done_params | set([need[0]])
        if need[0] == checking:
            for param in checking_range:
                tmp_param_values = param_values_list + [param]
                to_return_dict.update(get_slope_intercept(n, f, checking, checking_range, names, all_dict,
                                                          other_params=other_params,
                                                          checking_val=param,
                                                          done_params=new_done_params,
                                                          param_values_list=tmp_param_values))
        else:
            for param in other_params[need[0]]:
                tmp_param_values = param_values_list + [param]
                to_return_dict.update(get_slope_intercept(n, f, checking, checking_range, names, all_dict,
                                                          other_params=other_params,
                                                          checking_val=checking_val,
                                                          done_params=new_done_params,
                                                          param_values_list=tmp_param_values))
        return to_return_dict

    wanted = tuple(param_values_list)

    if len(param_values_list) < max(2,len(names)):
        raise ValueError("{0}: Too few members of param_values_list ({1!r})".format(n,param_values_list))

    if wanted in all_dict:
        slope, intercept, low, high = all_dict[wanted]
    else:
        slope, intercept, low, high = do_theilsen([f(i,*param_values_list) for i in x],x)
        all_dict[wanted] = [slope, intercept, low, high]
    param_dict = dict(zip(names, param_values_list))
    other_values = tuple([param_dict[a] for a in names if (a != checking)])
    to_return_dict[tuple([other_values, checking_val])] = [slope, intercept, low, high]
    return to_return_dict
    
            
mps = MultiParameterSet('activation')
afs = ActivationFunctionSet(mps)
for n in sorted(iterkeys(mps.multiparam_func_dict['activation'])):
    if n in ('hat_gauss_rectangular', 'wave', 'multiparam_gauss'):
        continue
    mpf = mps.multiparam_func_dict['activation'][n]
    f = mpf.user_func
    params_check = {}
    other_params = {}
    params_gaussian = set([])
    for param_name in mpf.evolved_param_names:
        param_dict = mpf.evolved_param_dicts[param_name]
        min_value = param_dict.get('min_init_value', param_dict['min_value'])
        max_value = param_dict.get('max_init_value', param_dict['max_value'])
        init_type = param_dict.get('init_type', 'uniform')
        if init_type.lower() in 'uniform':
            params_check[param_name] = [min_value,max_value]
            other_params[param_name] = np.linspace(min_value,max_value, 10)
        elif init_type.lower() in ('gaussian','normal'):
            params_gaussian.add(param_name)
            if ('init_mean' in param_dict) and ('init_stdev' in param_dict):
                middle = param_dict['init_mean']
                stdev = param_dict['init_stdev']
            else:
                middle = (min_value+max_value)/2.0
                stdev = abs(max_value-min_value)/math.sqrt(12.0)
            params_check[param_name] = [min_value, (middle-stdev), middle,
                                        (middle+stdev), max_value]
            other_params[param_name] = [(middle-stdev),middle,(middle+stdev)]
        else:
            raise ValueError(
                "{0}: Unknown init_type {1!r} for param_use '{2}'".format(
                    n, init_type, param_name))
    all_dict = {}
    for param_name in iterkeys(params_check):
        if (not FULL_TEST) and (param_name in ('tilt','width')) and (not len(params_gaussian)):
            continue
        print("{0}: Checking {1}".format(n,param_name))
        if len(other_params) == 1:
            if param_name in params_gaussian:
                num_outer = int(math.ceil(500.0/6.0))
                num_inner = 500 - (2*num_outer)
                param_range = (list(np.linspace(params_check[param_name][0],
                                                params_check[param_name][1],
                                                num_outer))
                               + list(np.linspace(params_check[param_name][1],
                                                  params_check[param_name][3],
                                                  num_inner))
                               + list(np.linspace(params_check[param_name][3],
                                                  params_check[param_name][4],
                                                  num_outer)))
            else:
                param_range = np.linspace(params_check[param_name][0],
                                          params_check[param_name][1],
                                          500)
            param_range_x3 = []
            slope_list = []
            intercept_list = []
            for param in param_range:
                slope, intercept, low, high = do_theilsen([f(i,param) for i in x],x)
                param_range_x3.extend([param,param,param])
                slope_list.extend([low,slope,high])
                intercept_list.append(intercept)
            slope_corr, slope_p_val = stats.spearmanr(param_range_x3, slope_list)
            print("{0}: {1} slope p-val {2:n}, correlation {3:n}".format(
                n, param_name, slope_p_val, slope_corr))
            intercept_corr, intercept_p_val = stats.spearmanr(param_range, intercept_list)
            print("{0}: {1} intercept p-val {2:n}, correlation {3:n}".format(
                n, param_name, intercept_p_val, intercept_corr))
            median_intercept = median2(intercept_list)
            print("{0}: {1} intercept range: {2:n}/{3:n}/{4:n}".format(
                n, param_name, min(intercept_list), median_intercept, max(intercept_list)))
            min_good_intercept = min(-2,median_intercept)
            max_good_intercept = max(2,median_intercept)
            median_slope = median2(slope_list)
            if ((min(slope_list) > median_intercept) or
                (max(slope_list) < median_intercept) or
                (min(intercept_list) > median_slope) or
                (max(intercept_list) < median_slope)):
                plt.figure(figsize=(4, 4))
                plt.plot([param_range_x3[i] for i in range(len(param_range_x3))],
                         [slope_list[i] for i in range(len(slope_list))],
                         'g-')
                plt.title("{0} for {1} (slope)".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(min(slope_list),max(slope_list))
                plt.gca().set_aspect(1)
                plt.savefig("slope-{0}.png".format(n))
                plt.close()
                if (max(intercept_list)-min(intercept_list)) > NORM_EPSILON:
                    plt.figure(figsize=(4, 4))
                    params_use = [param_range[i] for i in range(len(param_range)) if min_good_intercept <= intercept_list[i] <= max_good_intercept]
                    plt.plot(params_use,
                             [intercept_list[i] for i in range(len(intercept_list)) if min_good_intercept <= intercept_list[i] <= max_good_intercept],
                             'r-')
                    plt.title("{0} for {1} (intercept)".format(n, param_name))
                    plt.grid()
                    plt.xlim(math.floor(10*min(params_use))/10.0,
                             math.ceil(10*max(params_use))/10.0)
                    plt.autoscale(enable=True, axis='y')
                    plt.gca().set_aspect(1)
                    plt.savefig("intercept-{0}.png".format(n))
                    plt.close()
            else:
                plt.figure(figsize=(4, 4))
                plt.plot([param_range[i] for i in range(len(param_range))],
                         [intercept_list[i] for i in range(len(intercept_list))],
                         'r-', label='intercept')
                plt.plot([param_range_x3[i] for i in range(len(param_range_x3))],
                         [slope_list[i] for i in range(len(slope_list))],
                         'g-', label='slope')
                plt.title("{0} for {1}".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(min(min(slope_list),max(min_good_intercept,min(intercept_list))),
                         max(max(slope_list),min(max_good_intercept,max(intercept_list))))
                plt.legend()
                plt.gca().set_aspect(1)
                plt.savefig("slope_intercept-{0}.png".format(n))
                plt.close()
        else:
            num_points = 200
            if len(params_check) < 2:
                num_points = 500
            if param_name in params_gaussian:
                num_outer = int(math.ceil(num_points/6.0))
                num_inner = num_points - (2*num_outer)
                param_range = (list(np.linspace(params_check[param_name][0],
                                                params_check[param_name][1],
                                                num_outer))
                               + list(np.linspace(params_check[param_name][1],
                                                  params_check[param_name][3],
                                                  num_inner))
                               + list(np.linspace(params_check[param_name][3],
                                                  params_check[param_name][4],
                                                  num_outer)))
            else:
                param_range = np.linspace(params_check[param_name][0],
                                          params_check[param_name][1],
                                          num_points)
            param_value_dict = get_slope_intercept(n, f, param_name, param_range, mpf.evolved_param_names, all_dict, other_params)
            other_values_dict = {}
            for pair in iterkeys(param_value_dict):
                other_values = pair[0]
                param_value = pair[1]
                new_dict = {param_value: param_value_dict[pair]}
                if other_values in other_values_dict:
                    other_values_dict[other_values].update(new_dict)
                else:
                    other_values_dict[other_values] = new_dict
            slope_corr_list = []
            slope_p_val_list = []
            intercept_corr_list = []
            intercept_p_val_list = []
            param_x3_list = []
            slope_list = []
            param_list = []
            intercept_list = []
            for other_values in iterkeys(other_values_dict):
                param_vals1 = []
                slope_vals = []
                param_vals2 = []
                intercept_vals = []
                for param_val in other_values_dict[other_values]:
                    slope, intercept, low, high = other_values_dict[other_values][param_val]
                    param_vals1.extend([param_val,param_val,param_val])
                    slope_vals.extend([low,slope,high])
                    param_vals2.append(param_val)
                    intercept_vals.append(intercept)
                slope_corr, slope_p_val = stats.spearmanr(param_vals1, slope_vals)
                if slope_p_val < 0.25:
                    param_x3_list.extend(param_vals1)
                    slope_list.extend(slope_vals)
                intercept_corr, intercept_p_val = stats.spearmanr(param_vals2, intercept_vals)
                if intercept_p_val < 0.25:
                    param_list.extend(param_vals2)
                    intercept_list.extend(intercept_vals)
                slope_corr_list.append(slope_corr)
                slope_p_val_list.append(slope_p_val)
                intercept_corr_list.append(intercept_corr)
                intercept_p_val_list.append(intercept_p_val)
            ignored, slope_p_val = stats.combine_pvalues(slope_p_val_list)
            ignored, intercept_p_val = stats.combine_pvalues(intercept_p_val_list)
            lower_slope_corr = stats.scoreatpercentile(slope_corr_list,25)
            median_slope_corr = stats.scoreatpercentile(slope_corr_list,50)
            higher_slope_corr = stats.scoreatpercentile(slope_corr_list,75)
            print("{0}: {1} slope p-value {2:n}, corrs {3:n}/{4:n}/{5:n}/{6:n}/{7:n}".format(
                n, param_name, slope_p_val, min(slope_corr_list), lower_slope_corr,
                median_slope_corr, higher_slope_corr, max(slope_corr_list)))
            print("{0}: {1} slope - {2:n} good data points".format(n, param_name,
                                                                   (len(slope_list)/3.0)))
            median_intercept = median2(intercept_list)
            lower_int_corr = stats.scoreatpercentile(intercept_corr_list,25)
            median_int_corr = stats.scoreatpercentile(intercept_corr_list,50)
            higher_int_corr = stats.scoreatpercentile(intercept_corr_list,75)
            print("{0}: {1} intercept p-value {2:n}, corrs {3:n}/{4:n}/{5:n}/{6:n}/{7:n}".format(
                n, param_name, intercept_p_val, min(intercept_corr_list), lower_int_corr,
                median_int_corr, higher_int_corr, max(intercept_corr_list)))
            print("{0}: {1} intercept - {2:n} good data points".format(n, param_name,
                                                                       len(intercept_list)))
            print("{0}: {1} intercept range: {2:n}/{3:n}/{4:n}".format(
                n, param_name, min(intercept_list), median_intercept, max(intercept_list)))
            min_good_intercept = min(-2,median_intercept)
            max_good_intercept = max(2,median_intercept)
            median_slope = median2(slope_list)
            if ((min(slope_list) > median_intercept) or
                (max(slope_list) < median_intercept) or
                (min(intercept_list) > median_slope) or
                (max(intercept_list) < median_slope)):
                plt.figure(figsize=(4, 4))
                plt.plot([param_x3_list[i] for i in range(len(param_x3_list))],
                         [slope_list[i] for i in range(len(slope_list))],
                         'g-')
                plt.title("{0} for {1} (slope)".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(math.floor(10*min(slope_list))/10.0,
                         math.ceil(10*max(slope_list))/10.0)
                plt.gca().set_aspect(1)
                plt.savefig("slope-{0}-{1}.png".format(n,param_name))
                plt.close()
                if (max(intercept_list)-min(intercept_list)) > NORM_EPSILON:
                    plt.figure(figsize=(4, 4))
                    params_use = [param_list[i] for i in range(len(param_list)) if min_good_intercept <= intercept_list[i] <= max_good_intercept]
                    plt.plot(params_use,
                             [intercept_list[i] for i in range(len(intercept_list)) if min_good_intercept <= intercept_list[i] <= max_good_intercept],
                             'r-')
                    plt.title("{0} for {1} (intercept)".format(n, param_name))
                    plt.grid()
                    plt.xlim(math.floor(10*min(params_use))/10.0,
                             math.ceil(10*max(params_use))/10.0)
                    plt.autoscale(enable=True, axis='y')
                    plt.gca().set_aspect(1)
                    plt.savefig("intercept-{0}-{1}.png".format(n, param_name))
                    plt.close()
            else:
                plt.figure(figsize=(4, 4))
                plt.plot([param_list[i] for i in range(len(param_list))],
                         [intercept_list[i] for i in range(len(intercept_list))],
                         'r-', label='intercept')
                plt.plot([param_x3_list[i] for i in range(len(param_x3_list))],
                         [slope_list[i] for i in range(len(slope_list))],
                         'g-', label='slope')
                plt.title("{0} for {1}".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(math.floor(10*min(min(slope_list),max(min_good_intercept,min(intercept_list))))/10.0,
                         math.ceil(10*max(max(slope_list),min(max_good_intercept,max(intercept_list))))/10.0)
                plt.legend()
                plt.gca().set_aspect(1)
                plt.savefig("slope_intercept-{0}-{1}.png".format(n,param_name))
                plt.close()
