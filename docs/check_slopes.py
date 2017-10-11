from __future__ import print_function, division

import itertools
import math
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats, interpolate

from corrstats import independent_corr

from neat.multiparameter import MultiParameterSet
from neat.activations import ActivationFunctionSet, log1p_activation
from neat.math_util import median2, NORM_EPSILON, tmean
from neat.six_util import iterkeys

MIN_P_VALUE = min(sys.float_info.epsilon, math.sqrt(sys.float_info.min))

FULL_TEST = True

DO_ONLY = ("mexican_hat", "rational_quadratic")

x_set = set(list(np.linspace(-2, -1, 100)) + list(np.linspace(-1, 1, 300)) + list(np.linspace(1, 2, 100)))
x = sorted(x_set)

print("Initially have {0:n} points in x".format(len(x)))

def get_chebyshev_inner(N):
    if N <= 1:
        return [0.0]
    return [math.cos(num*math.pi/(N-1)) for num in range(N)]

def get_chebyshev(N, min_max_range=None):
    init_points = get_chebyshev_inner(N)
    if min_max_range is None:
        return init_points
    min_num, max_num = min_max_range
    if min_num >= max_num:
        raise ValueError("Min_num {0!r} max_num {1!r}".format(min_num,max_num))
    mult = max_num-min_num
    return [min(max_num,max(min_num,(min_num+(mult*((num+1.0)/2.0))))) for num in init_points]

chebyshev_points_set = set(get_chebyshev_inner(9))
for num in get_chebyshev_inner(8):
    if num < 0.0:
        chebyshev_points_set.add(num-1.0)
    elif num > 0.0:
        chebyshev_points_set.add(num+1.0)
##chebyshev_points_set |= set([-2.0,-1.0,0.0,1.0,2.0])
chebyshev_points_set |= set([-1.0,0.0,1.0])
chebyshev_points = sorted(chebyshev_points_set)

for num in chebyshev_points:
    if num not in x:
        x.append(num)
x.sort()
print("Now have {0:n} points in x".format(len(x)))

test_low_listA = []
test_low_listB = []
test_low_listC = []
test_high_listA = []
test_high_listB = []
test_high_listC = []
for x_use in x:
    if -1.5 <= x_use <= 1.5:
        if x_use < -1*NORM_EPSILON:
            test_low_listB.append(x_use)
            if x_use > -0.5:
                test_low_listA.append(x_use)
            else:
                test_low_listA.extend([x_use,x_use])
                test_low_listC.append(x_use)
        elif x_use > NORM_EPSILON:
            test_high_listB.append(x_use)
            if x_use < 0.5:
                test_high_listA.append(x_use)
            else:
                test_high_listA.extend([x_use,x_use])
                test_high_listC.append(x_use)
DIV_FOR_SLOPE2A = (math.fsum(test_high_listA)/len(test_high_listA)) - (math.fsum(test_low_listA)/len(test_low_listA))
DIV_FOR_SLOPE2B = (math.fsum(test_high_listB)/len(test_high_listB)) - (math.fsum(test_low_listB)/len(test_low_listB))
DIV_FOR_SLOPE2C = (math.fsum(test_high_listC)/len(test_high_listC)) - (math.fsum(test_low_listC)/len(test_low_listC))
print("DIV_FOR_SLOPE2 is {0:n}/{1:n}/{2:n}".format(DIV_FOR_SLOPE2A, DIV_FOR_SLOPE2B, DIV_FOR_SLOPE2C))

def check_linearity(y_data, x_data):
    x_data_non_chebyshev = [num for num in x_data if num not in chebyshev_points_set]
    y_data_non_chebyshev = [y_data[i] for i in range(len(y_data)) if x_data[i] not in chebyshev_points_set]
    x_data_chebyshev = [num for num in x_data if num in chebyshev_points_set]
    y_data_chebyshev = [y_data[i] for i in range(len(y_data)) if x_data[i] in chebyshev_points_set]

    base_interp = interpolate.interp1d(x_data_chebyshev, y_data_chebyshev, kind='zero')
    linear_interp = interpolate.interp1d(x_data_chebyshev, y_data_chebyshev, kind='slinear')
    #print("linear_interp: {0!r}".format(linear_interp))
    #print("x_data_chebyshev: {0!r}".format(x_data_chebyshev))
    cubic_interp = interpolate.interp1d(x_data_chebyshev, y_data_chebyshev, kind='cubic')
    base_interp_result = base_interp(x_data_non_chebyshev)
    linear_interp_result = linear_interp(x_data_non_chebyshev)
    cubic_interp_result = cubic_interp(x_data_non_chebyshev)
    base_diffs = []
    linear_diffs = []
    cubic_diffs = []
    for i in range(len(x_data_non_chebyshev)):
        base_diffs.append(abs(base_interp_result[i]-y_data_non_chebyshev[i]))
        linear_diffs.append(abs(linear_interp_result[i]-y_data_non_chebyshev[i]))
        cubic_diffs.append(abs(cubic_interp_result[i]-y_data_non_chebyshev[i]))
    avg_base_diff = math.fsum(base_diffs)/len(base_diffs)
    avg_linear_diff = math.fsum(linear_diffs)/len(linear_diffs)
    avg_cubic_diff = math.fsum(cubic_diffs)/len(cubic_diffs)
    if avg_base_diff <= min(avg_linear_diff,avg_cubic_diff):
        return 0.5
    if avg_linear_diff <= avg_cubic_diff:
        return 1.0
    if avg_linear_diff >= avg_base_diff:
        return 0.0
    return (avg_base_diff-avg_linear_diff)/(avg_base_diff-avg_cubic_diff)
    
##    ignored_slope, ignored_intercept, rvalue_linear, ignored_pvalue, ignored_stderr = stats.linregress(x_data, y_data)
##    log1p_x_data = [log1p_activation(a/math.exp(0.5)) for a in x_data]
##    log1p_y_data = [log1p_activation(b/math.exp(0.5)) for b in y_data]
##    ignored_slope, ignored_intercept, rvalue_loglog, ignored_pvalue, ignored_stderr = stats.linregress(log1p_x_data, log1p_y_data)
##    ignored_slope, ignored_intercept, rvalue_linlog, ignored_pvalue, ignored_stderr = stats.linregress(x_data, log1p_y_data)
##    ignored_slope, ignored_intercept, rvalue_loglin, ignored_pvalue, ignored_stderr = stats.linregress(log1p_x_data, y_data)
##    best_nonlin = max(rvalue_loglog, rvalue_linlog, rvalue_loglin, key=abs)
##    if max(abs(rvalue_linear),abs(best_nonlin)) >= 1.0:
##        if abs(rvalue_linear) > abs(best_nonlin):
##            return 1.0
##        if abs(rvalue_linear) < abs(best_nonlin):
##            return 0.0
##        return 0.5
##    ignored_z, pvalue = independent_corr(rvalue_linear, best_nonlin, n=len(x_data), twotailed=False)
##    if not (0.0 <= pvalue <= 0.5):
##        raise ValueError("Got back pvalue of {0!r} for {1:n} vs {2:n} ({3:n} data points)".format(
##            pvalue, rvalue_linear, best_nonlin, len(x_data)))
##    if abs(rvalue_linear) > abs(best_nonlin):
##        pvalue = 1.0-pvalue
##    return pvalue

def do_theilsen2(x_data, y_data):
    slope, ignored_intercept, ignored_low, ignored_high = stats.theilslopes(y_data, x_data, alpha=0.25)
    if math.isnan(slope):
        slope, ignored_intercept, ignored_rvalue, ignored_pvalue, ignored_stderr = stats.linregress(x_data, y_data)
    intercept_list = []
    for x_use, y_use in zip(x_data, y_data):
        intercept_list.append(y_use - (slope*x_use))
    intercept = tmean(intercept_list)
    return slope, intercept

def do_theilsen(y_data, x_data):
    slope, ignored_intercept, low, high = stats.theilslopes(y_data, x_data, alpha=0.25)
    intercept_list = []
    below_0_listA = []
    below_0_listB = []
    below_0_listC = []
    above_0_listA = []
    above_0_listB = []
    above_0_listC = []
    for x_use, y_use in zip(x_data,y_data):
        if not math.isnan(slope):
            intercept_list.append(y_use - (slope*x_use))
            if not (math.isnan(low) or math.isnan(high)):
                intercept_list.append(y_use - (low*x_use))
                intercept_list.append(y_use - (high*x_use))
        elif not (math.isnan(low) or math.isnan(high)):
            intercept_list.append(y_use - (low*x_use))
            intercept_list.append(y_use - (high*x_use))
        else:
            intercept_list.append(ignored_intercept)
        if -1.5 <= x_use <= 1.5:
            if x_use < -1*NORM_EPSILON:
                below_0_listB.append(y_use)
                if x_use > -0.5:
                    below_0_listA.append(y_use)
                else:
                    below_0_listA.extend([y_use,y_use])
                    below_0_listC.append(y_use)
            elif x_use > NORM_EPSILON:
                above_0_listB.append(y_use)
                if x_use < 0.5:
                    above_0_listA.append(y_use)
                else:
                    above_0_listA.extend([y_use,y_use])
                    above_0_listC.append(y_use)
    intercept = tmean(intercept_list)
    slope2A = (tmean(above_0_listA)-tmean(below_0_listA))/DIV_FOR_SLOPE2A
    slope2B = (tmean(above_0_listB)-tmean(below_0_listB))/DIV_FOR_SLOPE2B
    slope2C = (tmean(above_0_listC)-tmean(below_0_listC))/DIV_FOR_SLOPE2C
    linearity = check_linearity(y_data, x_data)
    return slope, intercept, low, high, stats.scoreatpercentile(y_data,1), stats.scoreatpercentile(y_data,99), linearity, slope2A, slope2B, slope2C

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
        slope, intercept, low, high, low_y, high_y, linearity, slope2A, slope2B, slope2C = all_dict[wanted]
    else:
        slope, intercept, low, high, low_y, high_y, linearity, slope2A, slope2B, slope2C = do_theilsen([f(i,*param_values_list) for i in x],x)
        all_dict[wanted] = [slope, intercept, low, high, low_y, high_y, linearity, slope2A, slope2B, slope2C]
    param_dict = dict(zip(names, param_values_list))
    other_values = tuple([param_dict[a] for a in names if (a != checking)])
    to_return_dict[tuple([other_values, checking_val])] = [slope, intercept, low, high, low_y, high_y, linearity, slope2A, slope2B, slope2C]
    return to_return_dict

def combine_pvalues(tmp_p_val_list, corr_list, which=None):
    if not tmp_p_val_list:
        return 1.0
    p_val_list = [max(MIN_P_VALUE,tmp_p_val_list[i]) for i in range(len(tmp_p_val_list))]
    corr_tmean = tmean(corr_list)
    if abs(corr_tmean) < NORM_EPSILON:
        if (min(p_val_list) < 0.0) and (max(p_val_list) > 0.0):
            return 1.0
        if not corr_tmean:
            ignored, p_val = stats.combine_pvalues(p_val_list)
            return p_val
    corr_median = median2(corr_list)
    if abs(corr_median) < NORM_EPSILON:
        if (min(p_val_list) < 0.0) and (max(p_val_list) > 0.0):
            return 1.0
        if not corr_median:
            ignored, p_val = stats.combine_pvalues(p_val_list)
            return p_val
    if math.copysign(1.0,corr_tmean) != math.copysign(1.0,corr_median):
        if which is not None:
            warnings.warn("{0}: Corr_tmean {1!r} vs corr_median {2!r}".format(which,corr_tmean,corr_median))
        else:
            warnings.warn("Corr_tmean {0!r} vs corr_median {1!r}".format(corr_tmean,corr_median))
        if (abs(corr_tmean) < NORM_EPSILON) and (abs(corr_median) >= NORM_EPSILON):
            corr_tmean = corr_median
    if corr_tmean > 0.0:
        use_p_val_list = [p_val_list[i] for i in range(len(p_val_list)) if corr_list[i] >= 0.0]
    else:
        use_p_val_list = [p_val_list[i] for i in range(len(p_val_list)) if corr_list[i] <= 0.0]
    ignored, p_val = stats.combine_pvalues(use_p_val_list)
    return p_val


mps = MultiParameterSet('activation')
afs = ActivationFunctionSet(mps)
for n in sorted(iterkeys(mps.multiparam_func_dict['activation']), reverse=True):
    ##if n in ('hat_gauss_rectangular', 'wave', 'multiparam_gauss'):
##        continue
    if (DO_ONLY is not None) and (n not in DO_ONLY):
        continue
    mpf = mps.multiparam_func_dict['activation'][n]
    f = mpf.user_func
    params_check = {}
    other_params = {}
    params_gaussian = set([])
    for param_name in mpf.evolved_param_names:
        param_dict = mpf.param_dicts[param_name].param_dict
        min_value = param_dict.get('min_init_value', param_dict['min_value'])
        max_value = param_dict.get('max_init_value', param_dict['max_value'])
        init_type = param_dict.get('init_type', 'uniform')
        if init_type.lower() in 'uniform':
            params_check[param_name] = [min_value,max_value]
            other_param_set = set(np.linspace(min_value,max_value,5))
            other_param_set |= set(get_chebyshev(2,min_max_range=(min_value,max_value)))
            other_params[param_name] = sorted(list(other_param_set))
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
            other_param_set = set([(middle-stdev),middle,(middle+stdev)])
            other_param_set |= set(get_chebyshev(2,min_max_range=(max((middle-(2.0*stdev),min_value)),
                                                                  min((middle+(2.0*stdev),max_value)))))
            other_params[param_name] = sorted(list(other_param_set))
        else:
            raise ValueError(
                "{0}: Unknown init_type {1!r} for param_use '{2}'".format(
                    n, init_type, param_name))
    all_dict = {}
    for param_name in iterkeys(params_check):
        if (not FULL_TEST) and (param_name in ('curve','g_curve','lower')):
            continue
        print("{0}: Checking {1}".format(n,param_name))
        if len(other_params) == 1:
            if param_name in params_gaussian:
                num_outer = 100
                num_inner = 400
                param_range = (list(np.linspace(params_check[param_name][0],
                                                params_check[param_name][1],
                                                num_outer))
                               + list(np.linspace(params_check[param_name][1],
                                                  params_check[param_name][3],
                                                  num_inner))
                               + list(np.linspace(params_check[param_name][3],
                                                  params_check[param_name][4],
                                                  num_outer)))
                param_range_set = set(param_range)
                param_range_set |= set(get_chebyshev(40,min_max_range=(params_check[param_name][1],
                                                                       params_check[param_name][3])))
                param_range_set |= set(get_chebyshev(10,min_max_range=(params_check[param_name][0],
                                                                       params_check[param_name][1])))
                param_range_set |= set(get_chebyshev(10,min_max_range=(params_check[param_name][3],
                                                                       params_check[param_name][4])))
                param_range = sorted(list(param_range_set))
            else:
                param_range_set = set(list(np.linspace(params_check[param_name][0],
                                                       params_check[param_name][1],
                                                       600)))
                param_range_set |= set(get_chebyshev(60,min_max_range=(params_check[param_name][0],
                                                                       params_check[param_name][1])))
                param_range = sorted(list(param_range_set))
            param_range_x3 = []
            param_list = []
            slope_list = []
            intercept_list = []
            low_y_list = []
            high_y_list = []
            linearity_list = []
            slope2_list = []
            for param in param_range:
                slope, intercept, low, high, low_y, high_y, linearity, slope2A, slope2B, slope2C = do_theilsen([f(i,param) for i in x],x)
                if not (math.isnan(slope)
                        or math.isnan(low)
                        or math.isnan(high)
                        or math.isnan(slope2A)
                        or math.isnan(slope2B)
                        or math.isnan(slope2C)):
                    param_range_x3.extend([param,param,param])
                    slope_list.extend([low,slope,high])
                    slope2_list.extend([slope2A,slope2B,slope2C])
                if not (math.isnan(intercept)
                        or math.isnan(low_y)
                        or math.isnan(high_y)
                        or math.isnan(linearity)):
                    param_list.append(param)
                    intercept_list.append(intercept)
                    low_y_list.append(low_y)
                    high_y_list.append(high_y)
                    linearity_list.append(linearity)
            if (not param_list) or (not param_range_x3):
                print("{0}: {1} All results nan?!?".format(n,param_name))
                continue
            slope_corr, slope_p_val = stats.spearmanr(param_range_x3, slope_list)
            print("{0}: {1} slope p-val {2:n}, correlation {3:n}".format(
                n, param_name, slope_p_val, slope_corr))
            print("{0}: {1} slope range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(slope_list), median2(slope_list), max(slope_list)))
            if ('tilt' in param_name) or ((slope_p_val <= 0.1) and (abs(slope_corr) > 0.25)):
                param_vs_slope_slope, param_vs_slope_intercept = do_theilsen2(param_range_x3, slope_list)
                print("{0}: slope = ({1!r}*{2}) + {3!r}".format(
                    n, param_vs_slope_slope, param_name, param_vs_slope_intercept))
                middle_slope_list = median2(slope_list)
                low_range = [param_range_x3[i] for i in range(len(param_range_x3)) if slope_list[i] <= middle_slope_list]
                param_vs_lslope_slope, param_vs_lslope_intercept = do_theilsen2(low_range,
                                                                                [slope_list[i] for i in range(len(slope_list)) if slope_list[i] <= middle_slope_list])
                print("{0}: below {1!r} slope = ({2!r}*{3}) + {4!r}".format(
                    n, middle_slope_list, param_vs_lslope_slope, param_name, param_vs_lslope_intercept))
                high_range = [param_range_x3[i] for i in range(len(param_range_x3)) if slope_list[i] >= middle_slope_list]
                param_vs_hslope_slope, param_vs_hslope_intercept = do_theilsen2(high_range,
                                                                                [slope_list[i] for i in range(len(slope_list)) if slope_list[i] >= middle_slope_list])
                print("{0}: above {1!r} slope = ({2!r}*{3}) + {4!r}".format(
                    n, middle_slope_list, param_vs_hslope_slope, param_name, param_vs_hslope_intercept))
                if param_vs_slope_slope >= 0.0:
                    print("{0}: Max {1} for low_range is {2!r}".format(n, param_name, max(low_range)))
                    print("{0}: Min {1} for high_range is {2!r}".format(n, param_name, min(high_range)))
                else:
                    print("{0}: Min {1} for low_range is {2!r}".format(n, param_name, min(low_range)))
                    print("{0}: Max {1} for high_range is {2!r}".format(n, param_name, max(high_range)))
                param_for_low_range = (middle_slope_list - param_vs_lslope_intercept)/param_vs_lslope_slope
                param_for_high_range = (middle_slope_list - param_vs_hslope_intercept)/param_vs_hslope_slope
                print("{0}: Alt thresholds for {1} are {2!r} and {3!r}".format(
                    n, param_name, param_for_low_range, param_for_high_range))
            intercept_corr, intercept_p_val = stats.spearmanr(param_list, intercept_list)
            print("{0}: {1} intercept p-val {2:n}, correlation {3:n}".format(
                n, param_name, intercept_p_val, intercept_corr))
            median_intercept = median2(intercept_list)
            print("{0}: {1} intercept range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(intercept_list), median_intercept, max(intercept_list)))
            if ((max(low_y_list) - min(low_y_list)) > NORM_EPSILON) or ((max(high_y_list)-min(high_y_list)) > NORM_EPSILON):
                low_y_corr, low_y_p_val = stats.spearmanr(param_list, low_y_list)
                high_y_corr, high_y_p_val = stats.spearmanr(param_list, high_y_list)
                print("{0}: {1} low_y p-val {2:n}, correlation {3:n}".format(
                    n, param_name, low_y_p_val, low_y_corr))
                print("{0}: {1} high_y p-val {2:n}, correlation {3:n}".format(
                    n, param_name, high_y_p_val, high_y_corr))
            print("{0}: {1} low_y/high_y range: {2:n}/{3!r}/{4!r}/{5:n}".format(
                n, param_name, min(low_y_list), median2(low_y_list),
                median2(high_y_list), max(high_y_list)))
            linearity_corr, linearity_p_val = stats.spearmanr(param_list, linearity_list)
            print("{0}: {1} linearity p-val {2:n}, correlation {3:n}".format(
                n, param_name, linearity_p_val, linearity_corr))
            print("{0}: {1} linearity range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(linearity_list), median2(linearity_list), max(linearity_list)))
            slope2_corr, slope2_p_val = stats.spearmanr(param_range_x3, slope2_list)
            print("{0}: {1} slope2 p-val {2:n}, correlation {3:n}".format(
                n, param_name, slope2_p_val, slope2_corr))
            print("{0}: {1} slope2 range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(slope2_list), median2(slope2_list), max(slope2_list)))
            if ('tilt' in param_name) or ((slope2_p_val <= 0.1) and (abs(slope2_corr) > 0.25)):
                param_vs_slope2_slope, param_vs_slope2_intercept = do_theilsen2(param_range_x3, slope2_list)
                print("{0}: slope2 = ({1!r}*{2}) + {3!r}".format(
                    n, param_vs_slope2_slope, param_name, param_vs_slope2_intercept))
            if ((linearity_p_val <= 0.1) or
                ((max(linearity_list)-min(linearity_list)) > NORM_EPSILON)):
                plt.figure(figsize=(4, 4))
                plt.scatter(param_list, linearity_list, c='b', marker='.')
                plt.title("{0} for {1} (linearity)".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(min(linearity_list),max(linearity_list))
                plt.savefig("linearity-{0}.png".format(n))
                plt.close()
            if ((slope2_p_val <= 0.1) or
                ((max(slope2_list)-min(slope2_list)) > NORM_EPSILON)):
                plt.figure(figsize=(4, 4))
                plt.scatter(param_range_x3, slope2_list, c='g', marker='.')
                plt.title("{0} for {1} (slope2)".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(min(slope2_list),max(slope2_list))
                plt.savefig("slope2-{0}.png".format(n))
                plt.close()
            min_good_intercept = min(-2,median_intercept)
            max_good_intercept = max(2,median_intercept)
            if slope_list:
                median_slope = median2(slope_list)
            else:
                median_slope = 0.0
            if ((not slope_list) or
                (min(slope_list) > median_intercept) or
                (max(slope_list) < median_intercept) or
                (min(intercept_list) > median_slope) or
                (max(intercept_list) < median_slope)):
                if slope_list:
                    plt.figure(figsize=(4, 4))
                    plt.scatter([param_range_x3[i] for i in range(len(param_range_x3))],
                                [slope_list[i] for i in range(len(slope_list))],
                                c='g',marker='.')
                    plt.title("{0} for {1} (slope)".format(n, param_name))
                    plt.grid()
                    plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                    plt.ylim(min(slope_list),max(slope_list))
                    plt.savefig("slope-{0}.png".format(n))
                    plt.close()
                if (max(intercept_list)-min(intercept_list)) > NORM_EPSILON:
                    plt.figure(figsize=(4, 4))
                    params_use = [param_list[i] for i in range(len(param_list)) if min_good_intercept <= intercept_list[i] <= max_good_intercept]
                    plt.scatter(params_use,
                                [intercept_list[i] for i in range(len(intercept_list)) if min_good_intercept <= intercept_list[i] <= max_good_intercept],
                                c='r', marker='.')
                    plt.title("{0} for {1} (intercept)".format(n, param_name))
                    plt.grid()
                    plt.xlim(math.floor(10*min(params_use))/10.0,
                             math.ceil(10*max(params_use))/10.0)
                    plt.ylim(max(min_good_intercept,min(intercept_list)),
                             min(max_good_intercept,max(intercept_list)))
                    #plt.gca().set_aspect(1)
                    plt.savefig("intercept-{0}.png".format(n))
                    plt.close()
            else:
                plt.figure(figsize=(4, 4))
                plt.plot([param_range[i] for i in range(len(param_range))],
                         [intercept_list[i] for i in range(len(intercept_list))],
                         'r-', label='intercept')
                plt.scatter([param_range_x3[i] for i in range(len(param_range_x3))],
                            [slope_list[i] for i in range(len(slope_list))],
                            c='g', marker='.')
                plt.title("{0} for {1}".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(min(min(slope_list),max(min_good_intercept,min(intercept_list))),
                         max(max(slope_list),min(max_good_intercept,max(intercept_list))))
                plt.legend()
                #plt.gca().set_aspect(1)
                plt.savefig("slope_intercept-{0}.png".format(n))
                plt.close()
        else:
            num_points = 240
            if len(params_check) < 2:
                num_points = 600
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
                param_range_set = set(param_range)
                param_range_set |= set(get_chebyshev(40,min_max_range=(params_check[param_name][1],
                                                                       params_check[param_name][3])))
                param_range_set |= set(get_chebyshev(10,min_max_range=(params_check[param_name][0],
                                                                       params_check[param_name][1])))
                param_range_set |= set(get_chebyshev(10,min_max_range=(params_check[param_name][3],
                                                                       params_check[param_name][4])))
                param_range = sorted(list(param_range_set))
            else:
                param_range_set = set(list(np.linspace(params_check[param_name][0],
                                                       params_check[param_name][1],
                                                       num_points)))
                param_range_set |= set(get_chebyshev(60,min_max_range=(params_check[param_name][0],
                                                                       params_check[param_name][1])))
                param_range = sorted(list(param_range_set))
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
            low_y_corr_list = []
            low_y_p_val_list = []
            high_y_corr_list = []
            high_y_p_val_list = []
            linearity_corr_list = []
            linearity_p_val_list = []
            slope2_corr_list = []
            slope2_p_val_list = []
            param_x3_list = []
            slope_list = []
            param_list = []
            intercept_list = []
            low_y_list = []
            high_y_list = []
            linearity_list = []
            slope2_list = []
            for other_values in iterkeys(other_values_dict):
                param_vals1 = []
                slope_vals = []
                param_vals2 = []
                intercept_vals = []
                low_y_vals = []
                high_y_vals = []
                linearity_vals = []
                slope2_vals = []
                for param_val in other_values_dict[other_values]:
                    slope, intercept, low, high, low_y, high_y, linearity, slope2A, slope2B, slope2C = other_values_dict[other_values][param_val]
                    param_vals1.extend([param_val,param_val,param_val])
                    slope_vals.extend([low,slope,high])
                    param_vals2.append(param_val)
                    intercept_vals.append(intercept)
                    low_y_vals.append(low_y)
                    high_y_vals.append(high_y)
                    linearity_vals.append(linearity)
                    slope2_vals.extend([slope2A,slope2B,slope2C])
                slope_corr, slope_p_val = stats.spearmanr(param_vals1, slope_vals)
                if math.isnan(slope_corr) or math.isnan(slope_p_val):
                    slope_corr = 0.0
                    slope_p_val = 1.0
                param_x3_list.extend(param_vals1)
                slope_list.extend(slope_vals)
                intercept_corr, intercept_p_val = stats.spearmanr(param_vals2, intercept_vals)
                if math.isnan(intercept_corr) or math.isnan(intercept_p_val):
                    intercept_corr = 0.0
                    intercept_p_val = 1.0
                param_list.extend(param_vals2)
                intercept_list.extend(intercept_vals)
                slope_corr_list.append(slope_corr)
                slope_p_val_list.append(slope_p_val)
                intercept_corr_list.append(intercept_corr)
                intercept_p_val_list.append(intercept_p_val)
                low_y_list.extend(low_y_vals)
                high_y_list.extend(high_y_vals)
                low_y_corr, low_y_p_val = stats.spearmanr(param_vals2, low_y_vals)
                if math.isnan(low_y_corr) or math.isnan(low_y_p_val):
                    low_y_corr = 0.0
                    low_y_p_val = 1.0
                high_y_corr, high_y_p_val = stats.spearmanr(param_vals2, high_y_vals)
                if math.isnan(high_y_corr) or math.isnan(high_y_p_val):
                    high_y_corr = 0.0
                    high_y_p_val = 1.0
                low_y_corr_list.append(low_y_corr)
                low_y_p_val_list.append(low_y_p_val)
                high_y_corr_list.append(high_y_corr)
                high_y_p_val_list.append(high_y_p_val)
                linearity_corr, linearity_p_val = stats.spearmanr(param_vals2, linearity_vals)
                if math.isnan(linearity_corr) or math.isnan(linearity_p_val):
                    linearity_corr = 0.0
                    linearity_p_val = 1.0
                linearity_corr_list.append(linearity_corr)
                linearity_p_val_list.append(linearity_p_val)
                linearity_list.extend(linearity_vals)
                slope2_corr, slope2_p_val = stats.spearmanr(param_vals1, slope2_vals)
                if math.isnan(slope2_corr) or math.isnan(slope2_p_val):
                    slope2_corr = 0.0
                    slope2_p_val = 1.0
                slope2_corr_list.append(slope2_corr)
                slope2_p_val_list.append(slope2_p_val)
                slope2_list.extend(slope2_vals)
            slope_p_val = combine_pvalues(slope_p_val_list, slope_corr_list, which='slope')
            intercept_p_val = combine_pvalues(intercept_p_val_list, intercept_corr_list, which='intercept')
            lower_slope_corr = stats.scoreatpercentile(slope_corr_list,25)
            median_slope_corr = stats.scoreatpercentile(slope_corr_list,50)
            higher_slope_corr = stats.scoreatpercentile(slope_corr_list,75)
            print("{0}: {1} slope p-value {2:n}, corrs {3:n}/{4!r}/{5!r}/{6!r}/{7:n}".format(
                n, param_name, slope_p_val, min(slope_corr_list), lower_slope_corr,
                median_slope_corr, higher_slope_corr, max(slope_corr_list)))
            print("{0}: {1} slope range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(slope_list), median2(slope_list), max(slope_list)))
            median_intercept = median2(intercept_list)
            lower_int_corr = stats.scoreatpercentile(intercept_corr_list,25)
            median_int_corr = stats.scoreatpercentile(intercept_corr_list,50)
            higher_int_corr = stats.scoreatpercentile(intercept_corr_list,75)
            print("{0}: {1} intercept p-value {2:n}, corrs {3:n}/{4!r}/{5!r}/{6!r}/{7:n}".format(
                n, param_name, intercept_p_val, min(intercept_corr_list), lower_int_corr,
                median_int_corr, higher_int_corr, max(intercept_corr_list)))
            print("{0}: {1} intercept range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(intercept_list), median_intercept, max(intercept_list)))
            low_y_p_val = combine_pvalues(low_y_p_val_list, low_y_corr_list, which='low_y')
            high_y_p_val = combine_pvalues(high_y_p_val_list, high_y_corr_list, which='high_y')
            lower_low_y_corr = stats.scoreatpercentile(low_y_corr_list,25)
            median_low_y_corr = stats.scoreatpercentile(low_y_corr_list,50)
            higher_low_y_corr = stats.scoreatpercentile(low_y_corr_list,75)
            lower_high_y_corr = stats.scoreatpercentile(high_y_corr_list,25)
            median_high_y_corr = stats.scoreatpercentile(high_y_corr_list,50)
            higher_high_y_corr = stats.scoreatpercentile(high_y_corr_list,75)
            print("{0}: {1} low_y p-value {2:n}, corrs {3:n}/{4!r}/{5!r}/{6!r}/{7:n}".format(
                n, param_name, low_y_p_val, min(low_y_corr_list), lower_low_y_corr,
                median_low_y_corr, higher_low_y_corr, max(low_y_corr_list)))
            median_low_y = median2(low_y_list)
            print("{0}: {1} low_y range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(low_y_list), median_low_y, max(low_y_list)))
            print("{0}: {1} high_y p-value {2:n}, corrs {3:n}/{4!r}/{5!r}/{6!r}/{7:n}".format(
                n, param_name, high_y_p_val, min(high_y_corr_list), lower_high_y_corr,
                median_high_y_corr, higher_high_y_corr, max(high_y_corr_list)))
            median_high_y = median2(high_y_list)
            print("{0}: {1} high_y range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(high_y_list), median_high_y, max(high_y_list)))
            linearity_p_val = combine_pvalues(linearity_p_val_list, linearity_corr_list, which='linearity')
            lower_linearity_corr = stats.scoreatpercentile(linearity_corr_list,25)
            median_linearity_corr = stats.scoreatpercentile(linearity_corr_list,50)
            higher_linearity_corr = stats.scoreatpercentile(linearity_corr_list,75)
            median_linearity = median2(linearity_list)
            print("{0}: {1} linearity p-value {2:n}, corrs {3:n}/{4!r}/{5!r}/{6!r}/{7:n}".format(
                n, param_name, linearity_p_val, min(linearity_corr_list), lower_linearity_corr,
                median_linearity_corr, higher_linearity_corr, max(linearity_corr_list)))
            print("{0}: {1} linearity range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(linearity_list), median_linearity, max(linearity_list)))
            if ((linearity_p_val <= 0.1) or
                ((max(linearity_list) - min(linearity_list)) >= NORM_EPSILON)):
                plt.figure(figsize=(4, 4))
                plt.scatter(param_list, linearity_list, c='b', marker='.')
                plt.title("{0} for {1} (linearity)".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(min(linearity_list),max(linearity_list))
                plt.savefig("linearity-{0}-{1}.png".format(n, param_name))
                plt.close()
            slope2_p_val = combine_pvalues(slope2_p_val_list, slope2_corr_list, which='slope2')
            lower_slope2_corr = stats.scoreatpercentile(slope2_corr_list,25)
            median_slope2_corr = stats.scoreatpercentile(slope2_corr_list,50)
            higher_slope2_corr = stats.scoreatpercentile(slope2_corr_list,75)
            median_slope2 = median2(slope2_list)
            print("{0}: {1} slope2 p-value {2:n}, corrs {3:n}/{4!r}/{5!r}/{6!r}/{7:n}".format(
                n, param_name, slope2_p_val, min(slope2_corr_list), lower_slope2_corr,
                median_slope2_corr, higher_slope2_corr, max(slope2_corr_list)))
            print("{0}: {1} slope2 range: {2:n}/{3!r}/{4:n}".format(
                n, param_name, min(slope2_list), median_slope2, max(slope2_list)))
            if ((slope2_p_val <= 0.1) or
                ((max(slope2_list) - min(slope2_list)) >= NORM_EPSILON)):
                plt.figure(figsize=(4, 4))
                plt.scatter(param_x3_list, slope2_list, c='g', marker='.')
                plt.title("{0} for {1} (slope2)".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(min(slope2_list),max(slope2_list))
                plt.savefig("slope2-{0}-{1}.png".format(n, param_name))
                plt.close()
            min_good_intercept = min(-2,median_intercept)
            max_good_intercept = max(2,median_intercept)
            if slope_list:
                median_slope = median2(slope_list)
            else:
                median_slope = 0.0
            if ((not slope_list) or
                (min(slope_list) > median_intercept) or
                (max(slope_list) < median_intercept) or
                (min(intercept_list) > median_slope) or
                (max(intercept_list) < median_slope)):
                if slope_list:
                    plt.figure(figsize=(4, 4))
                    plt.scatter([param_x3_list[i] for i in range(len(param_x3_list))],
                                [slope_list[i] for i in range(len(slope_list))],
                                c='g', marker='.')
                    plt.title("{0} for {1} (slope)".format(n, param_name))
                    plt.grid()
                    plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                    plt.ylim(math.floor(10*min(slope_list))/10.0,
                             math.ceil(10*max(slope_list))/10.0)
                    plt.savefig("slope-{0}-{1}.png".format(n,param_name))
                    plt.close()
                if (max(intercept_list)-min(intercept_list)) > NORM_EPSILON:
                    plt.figure(figsize=(4, 4))
                    params_use = [param_list[i] for i in range(len(param_list)) if min_good_intercept <= intercept_list[i] <= max_good_intercept]
                    plt.scatter(params_use,
                                [intercept_list[i] for i in range(len(intercept_list)) if min_good_intercept <= intercept_list[i] <= max_good_intercept],
                                c='r', marker='.')
                    plt.title("{0} for {1} (intercept)".format(n, param_name))
                    plt.grid()
                    plt.xlim(math.floor(10*min(params_use))/10.0,
                             math.ceil(10*max(params_use))/10.0)
                    plt.ylim(max(min_good_intercept,min(intercept_list)),
                             min(max_good_intercept,max(intercept_list)))
                    #plt.gca().set_aspect(1)
                    plt.savefig("intercept-{0}-{1}.png".format(n, param_name))
                    plt.close()
            else:
                plt.figure(figsize=(4, 4))
                plt.scatter([param_list[i] for i in range(len(param_list))],
                            [intercept_list[i] for i in range(len(intercept_list))],
                            c='r', marker='.')
                plt.scatter([param_x3_list[i] for i in range(len(param_x3_list))],
                            [slope_list[i] for i in range(len(slope_list))],
                            c='g', marker='.')
                plt.title("{0} for {1}".format(n, param_name))
                plt.grid()
                plt.xlim(params_check[param_name][0], params_check[param_name][-1])
                plt.ylim(math.floor(10*min(min(slope_list),max(min_good_intercept,min(intercept_list))))/10.0,
                         math.ceil(10*max(max(slope_list),min(max_good_intercept,max(intercept_list))))/10.0)
                #plt.legend()
                #plt.gca().set_aspect(1)
                plt.savefig("slope_intercept-{0}-{1}.png".format(n,param_name))
                plt.close()
