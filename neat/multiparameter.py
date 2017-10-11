"""
Enables the use of activation and aggregation functions
with, as well as the usual input, one or more evolvable numeric parameters.
"""
from __future__ import division

import copy
import functools
##import sys
import types
import warnings

from math import sqrt
from pprint import saferepr

try:
    from collections import MutableMapping, Sequence, Iterable, Hashable
except ImportError:
    from collections.abc import (MutableMapping, # pylint: disable=no-name-in-module,import-error
                                 Sequence,
                                 Iterable,
                                 Hashable)

from neat.attributes import FloatAttribute, BoolAttribute
from neat.math_util import NORM_EPSILON, tmean, mean
from neat.repr_util import repr_extract_function_name
from neat.six_util import iteritems, iterkeys

class _ParamValues(MutableMapping):
    def __init__(self, # pylint: disable=super-init-not-called
                 shared_names,
                 evolved_param_names,
                 param_namespace=None,
                 **param_values):
        self.param_values = dict(param_values)
        self.param_namespace = param_namespace
        self.shared_names = shared_names
        self.evolved_param_names = evolved_param_names
        #super(_ParamValues, self).__init__() # NEEDED/ADVISABLE?

    def set_param_namespace(self, param_namespace):
        self.param_namespace = param_namespace

    def __str__(self): # just has (potentially) used dict members
        if self.param_namespace is not None:
            namespace_dict = vars(self.param_namespace)
            tmp_dict = {}
            for name in self.evolved_param_names:
                if name in namespace_dict:
                    tmp_dict[name] = namespace_dict[name]
            param_namespace_repr = "Namespace({0!r})".format(tmp_dict)
        else:
            param_namespace_repr = 'None'

        shared_names_tmp = {}
        for name in self.evolved_param_names:
            if name in self.shared_names:
                shared_names_tmp[name] = self.shared_names[name]
        return "_ParamValues({0!r}, {1!r}, {2!s}, **{3!r})".format(
            shared_names_tmp,
            self.evolved_param_names,
            param_namespace_repr,
            self.param_values)

    def __repr__(self):
        if self.param_namespace is not None:
            param_namespace_repr = "Namespace({0!r})".format(vars(self.param_namespace))
        else:
            param_namespace_repr = 'None'
        return "_ParamValues({0!r}, {1!r}, {2!s}, **{3!r})".format(
            self.shared_names,
            self.evolved_param_names,
            param_namespace_repr,
            self.param_values)

    def copy(self):
        return self.__copy__

    def __copy__(self):
        other = _ParamValues(shared_names=self.shared_names,
                             evolved_param_names=self.evolved_param_names[:],
                             param_namespace=None,
                             **copy.copy(self.param_values))
        return other

    def __deepcopy__(self, memo_dict):
        other = _ParamValues(shared_names=self.shared_names,
                             evolved_param_names=self.evolved_param_names[:],
                             param_namespace=None,
                             **copy.deepcopy(self.param_values, memo_dict))
        return other

    def init_value(self, config=None, name=None,
                   param_namespace=None):
        if (param_namespace is not None) and (self.param_namespace != param_namespace):
            self.set_param_namespace(param_namespace)
        if name is None:
            if config is None:
                raise ValueError("Need either name or config")
            else:
                name = config.param_name
        if name in self.shared_names:
            if self.param_namespace is None:
                raise RuntimeError(
                    "Lack param_namespace to set {0} in".format(name))
            value = self.shared_names[name].attribute.init_value(self.shared_names[name])
            setattr(self.param_namespace, name, value)
            return value

        if name not in self.evolved_param_names:
            raise ValueError(
                "Name {0} is not in evolved_param_names".format(saferepr(name)))
        if config is None:
            raise ValueError(
                "Need config for initialization of {0}".format(name))

        self.param_values[name] = config.attribute.init_value(config)
        return self.param_values[name]

    def mutate_value(self, old_value=None, config=None, name=None,
                     param_namespace=None):
        """Mutates parameters, either shared or not."""
        if (param_namespace is not None) and (self.param_namespace != param_namespace):
            self.set_param_namespace(param_namespace)
        if name is None:
            if config is None:
                raise ValueError("Need either name or config")
            else:
                name = config.param_name
        if name in self.shared_names:
            if self.param_namespace is None:
                raise RuntimeError(
                    "Lack param_namespace to set {0} in".format(name))
            if not hasattr(self.param_namespace, name): # warn?
                old_value = self.shared_names[name].attribute.init_value(self.shared_names[name])
            elif old_value is None:
                old_value = getattr(self.param_namespace, name)
            value = self.shared_names[name].attribute.mutate_value(old_value,
                                                                   self.shared_names[name])
            setattr(self.param_namespace, name, value)
            return value

        if name not in self.evolved_param_names:
            raise KeyError(
                "Name {0} is not in evolved_param_names".format(saferepr(name)))
        if config is None:
            raise ValueError(
                "Need config for mutation of {0}".format(name))

        if name not in self.param_values:
            old_value = config.attribute.init_value(config) # warn?
        elif old_value is None:
            old_value = self.param_values[name]

        self.param_values[name] = config.attribute.mutate_value(old_value,
                                                                config)
        return self.param_values[name]

    def __getitem__(self, name):
        if name in self.shared_names:
            if self.param_namespace is None:
                raise RuntimeError(
                    "Lack param_namespace to get {0} from".format(name))
            if hasattr(self.param_namespace, name):
                return getattr(self.param_namespace, name)
            value = self.shared_names[name].attribute.init_value(self.shared_names[name])
            setattr(self.param_namespace, name, value)
            return value

        if name in self.param_values:
            return self.param_values[name]

        raise KeyError("Unknown name {0!s}".format(saferepr(name)))

    def __contains__(self, name):
        if name in self.param_values:
            return True
        if self.param_namespace is None: # can't init
            return False
        if name not in self.shared_names: # can't init
            return False
        return bool(name in self.evolved_param_names)

    def _get_used_names(self):
        return [name for name in self.shared_names if name in self.evolved_param_names]

    def __len__(self):
        num = len(self.param_values)
        if self.param_namespace is not None:
            num += len(self._get_used_names())
        return num

    def __setitem__(self, name, value):
        if name in self.shared_names:
            if self.param_namespace is None:
                raise RuntimeError(
                    "Lack param_namespace to set {0} in".format(name))
            # note: may wish to put back in validate & check for if in range/whatever
            setattr(self.param_namespace, name, value)
            return

        if name not in self.evolved_param_names:
            raise KeyError(
                "Name {0} is not in evolved_param_names".format(saferepr(name)))

        self.param_values[name] = value

    def __delitem__(self, name):
        if name in self.shared_names:
            raise KeyError( # will stop .clear()
                "Cannot delete shared_name {0}".format(name))
        if name in self.param_values:
            del self.param_values[name]
        else:
            raise KeyError("Unknown name {0}".format(saferepr(name)))

    def __iter__(self):
        if self.param_namespace is None:
            return iterkeys(self.param_values)
        # values come first so can be removed with .clear()
        return iter(list(self.param_values.keys()) + list(self._get_used_names()))

def _make_partial(user_func, name=None, **params):
    partial = functools.partial(user_func, **params)
    if name is not None:
        setattr(partial, '__name__', name)
    elif hasattr(user_func, '__name__'):
        setattr(partial, '__name__', user_func.__name__)
    if hasattr(user_func, '__doc__'):
        setattr(partial, '__doc__', user_func.__doc__)
    if hasattr(user_func, '__module__') and (user_func.__module__ is not None):
        setattr(partial, '__module__', user_func.__module__)
    else: # pragma: no cover
        setattr(partial, '__module__', _make_partial.__module__)
    return partial

##def _create_name(param_name, name=None, shared_names=None):
##    if (name is None) and (shared_names is None):
##        raise ValueError(
##            "{0}: Must have either name or shared_names".format(saferepr(param_name)))
##    if (shared_names is not None) and (param_name in shared_names):
##        return param_name
##    if name is not None:
##        return "{0}_{1}".format(name, param_name)
##    warnings.warn("Unable to determine form for {0}".format(param_name))
##    return param_name

class EvolvedMultiParameterFunction(object):
    """
    Holds, initializes, and mutates the evolved parameters for one instance
    of a multiparameter function.
    """
    def __init__(self, name, multi_param_func, do_values=True, param_namespace=None):
        self.name = name
        assert isinstance(multi_param_func,
                          MultiParameterFunction), "MPF {0!s} type '{1!s}' ({2!s})".format(
            name, type(multi_param_func), saferepr(multi_param_func))
        self.multi_param_func = multi_param_func
        self.user_func = multi_param_func.user_func
        self.evolved_param_names = multi_param_func.evolved_param_names
        self.param_dicts = multi_param_func.param_dicts
        self.param_namespace = param_namespace
        self.initialized = False
        self.shared_names = multi_param_func.shared_names

        if do_values:
            self.current_param_values = _ParamValues(shared_names=self.shared_names,
                                                     evolved_param_names=self.evolved_param_names,
                                                     param_namespace=self.param_namespace)
            self.init_value() # will only do ones possible without param_namespace if is None
        else:
            self.current_param_values = _ParamValues(shared_names=self.shared_names,
                                                     evolved_param_names=self.evolved_param_names,
                                                     param_namespace=None)

    @property
    def instance_name(self):
        value_dict = self.get_values()
        value_list = []
        for n in self.evolved_param_names:
            if n in value_dict:
                value_list.append(str(value_dict[n]))
            else:
                value_list.append('None')
        return self.name + '(' + ','.join([self.user_func.__code__.co_varnames[0]]
                                          + value_list) + ')'

    def set_param_namespace(self, param_namespace):
        self.param_namespace = param_namespace
        self.current_param_values.set_param_namespace(self.param_namespace)
        if not self.initialized:
            self.init_value(shared_only=True)

    def init_value(self, ignored_config=None, shared_only=False,
                   param_namespace=None):
        if (param_namespace is not None) and (self.param_namespace != param_namespace):
            self.set_param_namespace(param_namespace)
        if self.param_namespace is not None:
            for n in self.evolved_param_names:
                if (n in self.shared_names) or (not shared_only):
                    self.current_param_values.init_value(config=self.param_dicts[n])
            self.initialized = True
        elif not shared_only:
            for n in self.evolved_param_names:
                if n not in self.shared_names:
                    self.current_param_values.init_value(config=self.param_dicts[n])
        else:
            raise RuntimeError(
                "Cannot do initialization of shared names without param_namespace")

    def mutate_value(self, ignored_config=None,
                     param_namespace=None):
        if (param_namespace is not None) and (self.param_namespace != param_namespace):
            self.set_param_namespace(param_namespace)
        for n in self.evolved_param_names:
            self.current_param_values.mutate_value(old_value=self.current_param_values[n],
                                                   config=self.param_dicts[n])

    def set_values(self, **param_values): # TEST NEEDED!
        for n, val in iteritems(param_values):
            if n in self.evolved_param_names:
                self.current_param_values[n] = val
            else:
                raise LookupError(
                    "Parameter name {0!s} (val {1!s}) not among known {2!r} for {3!s}".format(
                        saferepr(n), saferepr(val), self.evolved_param_names, self.name))

    def get_values(self, n=None, do_copy=True, include_shared=True):
        if n is not None:
            return self.current_param_values[n]
        if do_copy:
            value_dict = {}
            if (self.param_namespace is not None) and include_shared:
                for n in self.evolved_param_names:
                    value_dict[n] = self.current_param_values[n]
            else:
                for n in self.evolved_param_names:
                    if n not in self.shared_names:
                        value_dict[n] = self.current_param_values[n]
            return value_dict
        return self.current_param_values

    def __str__(self):
        return self.instance_name

    def __repr__(self):
        value_dict = self.get_values()
        if self.param_namespace is not None:
            part1 = "EvolvedMultiParameterFunction({0!r}, {1!r})".format(
                self.name, self.multi_param_func)
            namespace_repr = "Namespace({0!r})".format(vars(self.param_namespace))
            part2 = ".set_values({0!r}).set_param_namespace({1!s})".format(
                value_dict,
                namespace_repr)
            return part1 + part2
        if value_dict:
            return "EvolvedMultiParameterFunction({0!r}, {1!r}).set_values({2!r})".format(
                self.name, self.multi_param_func, value_dict)
        return "EvolvedMultiParameterFunction({0!r}, {1!r})".format(
            self.name, self.multi_param_func)

    def distance(self, other, split=False):
        """Determine distances between two activation or aggregation functions."""
        categorical, continuous = self._distance_inner(other)
        if split:
            return [0.0,categorical,continuous] # structural, categorical, continuous
        else:
            return categorical+continuous

    def _distance_inner(self, other):
        if not isinstance(other, EvolvedMultiParameterFunction):
            if ((other in self.name) or
                (self.name in other) or
                ((self.name == 'wave') and (other == 'sin'))): # approximate! XXX
                total_diff = 0.5 + len(self.evolved_param_names)
                return [total_diff/(1+len(self.evolved_param_names)), 0.0]
            return [1.0,0.0]

        if (self.name == other.name) and (self.instance_name == other.instance_name):
            return [0.0,0.0]

        diffs_cat = []
        diffs_cont = []

        if self.name != other.name:
            diffs_cat.append(1.0)
        else:
            diffs_cat.append(0.0)
        diffs_cont.append(0.0)

        for n in self.evolved_param_names:
            if n not in other.evolved_param_names:
                diffs_cat.append(1.0)
                diffs_cont.append(0.0)
                continue
            param_dict = self.param_dicts[n].param_dict
            if param_dict['param_type'] in ('float', 'int'):
                diffs_cat.append(0.0)
                diff = abs(self.current_param_values[n] -
                           other.current_param_values[n])
                if diff:
                    div_by = max(NORM_EPSILON,
                                 abs(param_dict['max_value'] - param_dict['min_value']))
                    this_diff = diff / div_by
                    if this_diff > 1.0: # pragma: no cover
                        raise RuntimeError(
                            "{0} for {1}: This_diff {2:n} > 1.0 (diff {3:n}, div_by {4:n})".format(
                                self.name, n, this_diff, diff, div_by))
                    diffs_cont.append(this_diff)
                else:
                    diffs_cont.append(0.0)
            elif param_dict['param_type'] == 'bool':
                if self.current_param_values[n] != other.current_param_values[n]:
                    diffs_cont.append(0.5)
                    diffs_cat.append(0.5)
                else:
                    diffs_cont.append(0.0)
                    diffs_cat.append(0.0)
            else:
                raise ValueError("Unknown what to do with param_type {0!s} for {1!s}".format(
                    saferepr(param_dict['param_type']), self.name))
        if self.name != other.name: # to make symmetrical
            for n in other.evolved_param_names:
                if n not in self.evolved_param_names:
                    diffs_cat.append(1.0)
                    diffs_cont.append(0.0)
        return [mean(diffs_cat),mean(diffs_cont)]

    def copy(self):
        return copy.copy(self)

    def __copy__(self):
        other = EvolvedMultiParameterFunction(self.name[:],
                                              copy.copy(self.multi_param_func),
                                              do_values=False)
        other.set_values(**self.get_values(include_shared=False))
        return other

    def __deepcopy__(self, memo_dict):
        other = EvolvedMultiParameterFunction(self.name[:],
                                              copy.deepcopy(self.multi_param_func,memo_dict),
                                              do_values=False)
        other.set_values(**self.get_values(include_shared=False))
        return other

    def get_func(self):
        return _make_partial(self.user_func, self.instance_name, **self.current_param_values)

class DefaultParamInit(object):
    """
    Initializes the configuration for a parameter with default values,
    and holds the configuration and configured Attribute.
    """
    def __init__(self,
                 name,
                 param_name,
                 full_init_defaults=True,
                 **evolved_param_dict):
        self.name = name
        self.param_name = param_name
        self.attribute = None
        self.param_dict = evolved_param_dict

        self.init_defaults(full=full_init_defaults)

    def _init_from_mean_stdev(self):
        if ('max_init_value' in self.param_dict) or ('min_init_value' in self.param_dict):
            warnings.warn(
                "{0}: Have init_mean/stdev - ignoring max/min_init_value".format(
                    self.name))
        if self.param_dict['init_type'].lower() in 'uniform':
            plus_minus = self.param_dict['init_stdev']*2.0
        elif self.param_dict['init_type'].lower() in ('gaussian','normal'):
            plus_minus = (self.param_dict['init_stdev']*sqrt(12.0))/2.0
        else:
            raise ValueError(
                "{0}: Unknown init_type {1}".format(
                    self.name, saferepr(self.param_dict['init_type'])))

        self.param_dict['max_init_value'] = (self.param_dict['init_mean']
                                             + plus_minus)
        self.param_dict['min_init_value'] = (self.param_dict['init_mean']
                                              - plus_minus)
        self.param_dict.setdefault('max_value',
                                   self.param_dict['max_init_value'])
        self.param_dict.setdefault('min_value',
                                   self.param_dict['min_init_value'])
        if self.param_dict['max_value'] < self.param_dict['max_init_value']:
            raise ValueError(
                "{0}: max_value {1:n} too low for init_mean, init_stdev".format(
                    self.name, self.param_dict['max_value']))
        if self.param_dict['min_value'] > self.param_dict['min_init_value']:
            raise ValueError(
                "{0}: min_value {1:n} too high for init_mean, init_stdev".format(
                    self.name, self.param_dict['min_value']))

    def _init_from_min_max(self):
        self.param_dict.setdefault('max_init_value',
                                   self.param_dict['max_value'])
        self.param_dict.setdefault('min_init_value',
                                   self.param_dict['min_value'])
        self.param_dict.setdefault('max_value',
                                   self.param_dict['max_init_value'])
        self.param_dict.setdefault('min_value',
                                   self.param_dict['min_init_value'])

        middle = (self.param_dict['max_init_value'] +
                  self.param_dict['min_init_value'])/2.0
        if ('init_mean' in self.param_dict) and (middle != self.param_dict['init_mean']):
            if (abs(middle-self.param_dict['init_mean']) <= NORM_EPSILON):
                warnings.warn("{0}: Using calculated value {1:n} for init_mean".format(
                    self.name, middle))
                self.param_dict['init_mean'] = middle
            else:
                raise ValueError(
                    "{0}: Contradictory max/min mean {1:n} and init_mean {2:n}".format(
                        self.name,middle,self.param_dict['init_mean']))
        else:
            self.param_dict.setdefault('init_mean', middle)
        for_stdev = abs(self.param_dict['max_init_value']-self.param_dict['min_init_value'])/4.0
        # actual standard deviation of uniform distribution is width/sqrt(12) -
        # use of 1/4 range in the uniform distribution FloatAttribute setup
        # (and thus the above) is to make it easier to figure out how to
        # get a given initialization range that is not the same as the
        # overall min/max range.
        if self.param_dict['init_type'].lower() in ('gaussian', 'normal'):
            for_stdev = (4.0*for_stdev)/sqrt(12.0)
        if ('init_stdev' in self.param_dict) and (for_stdev != self.param_dict['init_stdev']):
            if (abs(for_stdev-self.param_dict['init_stdev']) <= NORM_EPSILON):
                warnings.warn("{0}: Using calculated value {1:n} for init_stdev".format(
                    self.name,for_stdev))
                self.param_dict['init_stdev'] = for_stdev
            else:
                raise ValueError(
                    "{0}: Contradictory max/min stdev {1:n} and init_stdev {2:n}".format(
                        self.name,for_stdev,self.param_dict['init_stdev']))
        else:
            self.param_dict.setdefault('init_stdev', for_stdev)

    def init_defaults(self, full=True):
        """
        Initializes defaults for one parameter's attribute settings for a multiparameter
        function. Can also re-initialize after user settings changes, provided the user
        previously deletes any old settings that should be initialized.
        """
        self.param_dict.setdefault('param_type', 'float')

        if self.param_dict['param_type'] in ('float','int'):
            if full:
                self.param_dict.setdefault('init_type','uniform')

                if ('init_mean' in self.param_dict) and ('init_stdev' in self.param_dict):
                    self._init_from_mean_stdev()

                elif ((('max_init_value' in self.param_dict) or ('max_value' in self.param_dict))
                      and
                      (('min_init_value' in self.param_dict) or ('min_value' in self.param_dict))):
                    self._init_from_min_max()

                else:
                    raise ValueError(
                        "{0}: Have neither max/min nor init_mean+init_stdev".format(
                            self.name))

                # below here is mainly intended for users wanting to use built-in
                # multiparameter functions without too much initialization worries
                self.param_dict.setdefault('replace_rate', 0.1)
                mutate_rate = min((1.0-self.param_dict['replace_rate']),
                                  (self.param_dict['replace_rate']*5.0))
                self.param_dict.setdefault('mutate_rate', mutate_rate)
                if self.param_dict['mutate_rate'] > 0:
                    mutate_power = (min(1.0,(self.param_dict['replace_rate']/
                                             self.param_dict['mutate_rate']))*
                                    (abs(self.param_dict['max_init_value']
                                         -self.param_dict['min_init_value'])/
                                     sqrt(12.0)))
                    self.param_dict.setdefault('mutate_power', mutate_power)
                else: # pragma: no cover
                    self.param_dict.setdefault('mutate_power', 0.0)
                if (self.param_dict['replace_rate']+self.param_dict['mutate_rate']) < 0.05:
                    warnings.warn("{0}: Replace_rate {1!r}, mutate_rate {2!r}".format(
                        self.name, self.param_dict['replace_rate'],
                        self.param_dict['mutate_rate']))
                elif ((self.param_dict['replace_rate'] > 0.0)
                      and (self.param_dict['mutate_power'] < NORM_EPSILON)):
                    warnings.warn("{0}: Mutate_rate {1!r}, mutate_power {2!r}".format(
                        self.name, self.param_dict['mutate_rate'],
                        self.param_dict['mutate_power']))

            param_dict2 = copy.copy(self.param_dict)
            del param_dict2['param_type']
            del param_dict2['min_init_value']
            del param_dict2['max_init_value']

            self.attribute = FloatAttribute(name=self.name, # TODO: IntAttribute
                                             default_ok=True,
                                             **param_dict2)
        elif self.param_dict['param_type'] == 'bool':
            if full:
                self.param_dict.setdefault('default', 'random')
                rate_to_true = max(0.0,min(1.0,(self.param_dict.get('mutate_rate',0.0)
                                                +self.param_dict.get('rate_to_true_add',0.0))))
                rate_to_false = max(0.0,min(1.0,(self.param_dict.get('mutate_rate',0.0)
                                                 +self.param_dict.get('rate_to_false_add',0.0))))
                self.param_dict['mutate_rate'] = max(0.1,min(rate_to_true,rate_to_false))
                self.param_dict['rate_to_true_add'] = max(0.0,
                                                          (rate_to_true
                                                           -self.param_dict['mutate_rate']))
                self.param_dict['rate_to_false_add'] = max(0.0,
                                                           (rate_to_false
                                                            -self.param_dict['mutate_rate']))

            param_dict2 = copy.copy(self.param_dict)
            del param_dict2['param_type']

            self.attribute = BoolAttribute(name=self.name,
                                           default_ok=True,
                                           **param_dict2)
        else:
            raise ValueError(
                "Unknown param_type {0!s} for {1!s}".format(
                    saferepr(self.param_dict['param_type']), self.name))

        for x, y in iteritems(self.param_dict):
            # so that this can be used as a config for the attribute
            setattr(self, self.attribute.config_item_name(x), y)

    def __copy__(self):
        return DefaultParamInit(name=self.name[:],
                                param_name=self.param_name[:],
                                full_init_defaults=False,
                                **copy.copy(self.param_dict))

    def __deepcopy__(self, memo_dict):
        return DefaultParamInit(name=self.name[:],
                                param_name=self.param_name[:],
                                full_init_defaults=False,
                                **copy.deepcopy(self.param_dict, memo_dict))


class MultiParameterFunction(object):
    """Holds and initializes configuration information for one multiparameter function."""
    def __init__(self,
                 name,
                 which_type,
                 user_func,
                 evolved_param_names,
                 full_init_defaults=True,
                 shared_names=None,
                 **evolved_param_dicts):
        self.name = name + "_MPF"
        self.orig_name = name
        self.which_type = which_type # activation or aggregation
        self.user_func = user_func
        self.evolved_param_names = evolved_param_names # for the ordering
        self.evolved_param_dicts = evolved_param_dicts
        self.param_dicts = {}
        self.shared_names = shared_names

        for n in evolved_param_names:
            if n not in self.shared_names:
                tmp_name = "{0}_{1}".format(self.orig_name,n)
                self.param_dicts[n] = DefaultParamInit(name=tmp_name,
                                                       param_name=n,
                                                       full_init_defaults=full_init_defaults,
                                                       **self.evolved_param_dicts[n])
            else:
                self.param_dicts[n] = self.shared_names[n]

    def init_instance(self, param_namespace=None):
        return EvolvedMultiParameterFunction(self.orig_name,
                                             self,
                                             param_namespace=param_namespace)

    def copy_and_change(self,
                        del_not_changed=False,
                        del_param_dicts=None,
                        new_param_dicts=None):
        """
        Makes a copy of the MultiParameterFunction instance, does deletions and
        substitutions, initializes any remaining defaults, and returns the new instance.
        If del_not_changed is True, any values for parameters in new_param_dicts that are not
        set will be deleted and reinitialized. Will not alter shared names.
        """
        new = copy.deepcopy(self)
        n_done = set([])
        if del_param_dicts is not None:
            for n in iterkeys(del_param_dicts):
                if n in self.shared_names:
                    raise ValueError(
                        "Cannot delete shared name {0}".format(n))
                if isinstance(del_param_dicts[n], (Sequence,Iterable)):
                    for x in del_param_dicts[n]:
                        del new.evolved_param_dicts[n][x]
                elif del_param_dicts[n] is None:
                    param_type = new.evolved_param_dicts[n]['param_type']
                    del new.evolved_param_dicts[n]
                    new.evolved_param_dicts[n] = {'param_type':param_type}
                elif isinstance(del_param_dicts[n], Hashable):
                    del new.evolved_param_dicts[n][del_param_dicts[n]]
                else: # pragma: no cover
                    warnings.warn(
                        "Deleting all evolved_param_dicts[{0!r}] due to unhashable {1!r}".format(
                            n, del_param_dicts[n]),
                        RuntimeWarning)
                    param_type = new.evolved_param_dicts[n]['param_type']
                    del new.evolved_param_dicts[n]
                    new.evolved_param_dicts[n] = {'param_type':param_type}
                n_done.add(n)
        if new_param_dicts is not None:
            for n in iterkeys(new_param_dicts):
                if n in self.shared_names:
                    raise ValueError(
                        "Cannot change shared name {0} via copy_and_change".format(n))
                if del_not_changed:
                    param_type = new.evolved_param_dicts[n]['param_type']
                    del new.evolved_param_dicts[n]
                    new.evolved_param_dicts[n] = {'param_type':param_type}
                for x, y in iteritems(new_param_dicts[n]):
                    new.evolved_param_dicts[n][x] = y
                n_done.add(n)
        for n in n_done:
            tmp_name = "{0}_{1}".format(new.orig_name,n)
            new.param_dicts[n] = DefaultParamInit(name=tmp_name,
                                                  param_name=n,
                                                  full_init_defaults=True,
                                                  **new.evolved_param_dicts[n])
        return new

    def __repr__(self): # TEST NEEDED! Should be able to duplicate by using this as an init...
        to_return_list = []
        to_return_list.append('orig_name=' + repr(self.orig_name))
        to_return_list.append('which_type=' + repr(self.which_type))
        to_return_list.append('user_func=' + repr_extract_function_name(self.user_func,
                                                                        with_module=True))
        to_return_list.append('evolved_param_names=' + repr(self.evolved_param_names))

        for n in self.evolved_param_names:
            to_return_list.append(repr(n) + '=' + repr(self.param_dicts[n].param_dict))
        return 'MultiParameterFunction(' + ",".join(to_return_list) + ')'

    def __copy__(self):
        new_evolved_param_dicts = {}
        for n in self.evolved_param_names:
            new_evolved_param_dicts[n] = copy.copy(self.param_dicts[n].param_dict)
        return MultiParameterFunction(name=self.orig_name[:], which_type=self.which_type[:],
                                      user_func=copy.copy(self.user_func),
                                      evolved_param_names=self.evolved_param_names[:],
                                      full_init_defaults=False,
                                      shared_names=self.shared_names,
                                      **new_evolved_param_dicts)

    def __deepcopy__(self, memo_dict):
        new_evolved_param_dicts = {}
        for n in self.evolved_param_names:
            new_evolved_param_dicts[n] = copy.deepcopy(self.param_dicts[n].param_dict,memo_dict)
        return MultiParameterFunction(name=self.orig_name[:], which_type=self.which_type[:],
                                      user_func=copy.deepcopy(self.user_func, memo_dict),
                                      evolved_param_names=self.evolved_param_names[:],
                                      full_init_defaults=False,
                                      shared_names=self.shared_names,
                                      **new_evolved_param_dicts)


class BadFunctionError(Exception):
    pass

class InvalidFunctionError(TypeError, BadFunctionError):
    pass

class UnknownFunctionError(LookupError, BadFunctionError):
    pass

class MultiParameterSet(object):
    """
    Holds the set of (potentially multiparameter) functions
    and contains methods for dealing with them.
    """
    def __init__(self, *which_types):
        self.norm_func_dict = {}
        self.multiparam_func_dict = {}
        self.shared_names = {}
        for which_type in list(which_types):
            self.norm_func_dict[which_type] = {}
            self.multiparam_func_dict[which_type] = {}
            self.shared_names[which_type] = {}

    def add_shared_name(self, name, which_type, **kwargs):
        if len(name) < 2: # TEST NEEDED!
            raise ValueError("Min length of name ({0}) for add_shared_name is 2".format(
                saferepr(name)))
        other_types = [other_type for other_type in self.shared_names
                       if ((other_type != which_type) and
                           (name in self.shared_names[other_type]))]
        if other_types: # TEST NEEDED!
            raise ValueError(
                "Shared name conflict: {0} in {1!r} already".format(
                    name, other_types))
        self.shared_names[which_type][name] = DefaultParamInit(name=name,
                                                               param_name=name,
                                                               full_init_defaults=True,
                                                               **kwargs)

    def is_valid_func(self, name, which_type):
        if name in self.multiparam_func_dict[which_type]:
            return True
        if name in self.norm_func_dict[which_type]:
            return True
        if name.endswith(')'):
            try:
                ignored = self.get_func(name, which_type)
            except (BadFunctionError,LookupError):
                return False
            except RuntimeError:
                raise UnknownFunctionError("Called with uncertain name {0}".format(saferepr(name)))
            else:
                return True
        return False

    def is_multiparameter(self, name, which_type):
        if name in self.multiparam_func_dict[which_type]:
            return True
        if name.endswith(')'):
            try:
                maybe = self.get_func(name, which_type)
            except (BadFunctionError,LookupError):
                return False
            except RuntimeError:
                raise UnknownFunctionError("Called with uncertain name {0}".format(saferepr(name)))
            else:
                response = repr(maybe)
                return bool('partial' in response.lower())
        return False

    def init_multiparameter(self, name, instance, ignored_config=None, param_namespace=None):
        which_type = instance.name
        multiparam_func_dict = self.multiparam_func_dict[which_type]
        multiparam_func = multiparam_func_dict[name]
        return multiparam_func.init_instance(param_namespace=param_namespace)

    def get_MPF(self,
                name, # type: str
                which_type # type: str
                ):
        # type: (...) -> MultiParameterFunction
        """Fetches the named MultiParameterFunction instance."""

        if name in self.multiparam_func_dict[which_type]:
            mpfunc_dict = self.multiparam_func_dict[which_type] # type: Dict[str, MultiParameterFunction]
            # Allows for altering configuration - preferably via copy_and_change!
            return mpfunc_dict[name]
        raise UnknownFunctionError("Unknown {!s} MPF function {!r}".format(which_type,name))

    def _get_func_inner(self,
                        name, # type: str
                        which_type, # type: str
                        return_partial, # type: bool
                        param_namespace=None
                       ):
        if not name.endswith(')'):
            raise UnknownFunctionError("Unknown {!s} function {!s} - no end )".
                                       format(which_type,saferepr(name)))

        param_start = name.find('(')
        if param_start < 0:
            raise UnknownFunctionError("Unknown {!s} function {!s} - no start (".
                                       format(which_type,saferepr(name)))

        func_name = name[:param_start]
        if func_name not in self.multiparam_func_dict[which_type]:
            raise UnknownFunctionError("Unknown {0!s} function {1!s} (from {2!s})".
                                       format(which_type,saferepr(func_name),
                                              saferepr(name)))
        multiparam_func = self.multiparam_func_dict[which_type][func_name]

        param_values = name[(param_start+1):(len(name)-1)].split(',')

        # HAS TO DEAL WITH SHARED NAMES!

        if len(param_values) == (len(multiparam_func.evolved_param_names)+1):
            if param_values[0] in ('x','z',multiparam_func.user_func.__code__.co_varnames[0]):
                param_values = param_values[1:]

        if len(multiparam_func.evolved_param_names) < len(param_values):
            raise RuntimeError(
                "Too many ({0:n}) param_values in name {1!r} - should be max {2:n}".format(
                    len(param_values), name, len(multiparam_func.evolved_param_names)))
        elif len(multiparam_func.evolved_param_names) > len(param_values):
            warnings.warn(
                "{0!r}: Only {1:n} param_values, but function takes {2:n}".format(
                    name, len(param_values), len(multiparam_func.evolved_param_names)))

        init_params = dict(zip(multiparam_func.evolved_param_names, param_values))
        params = {}
        for n2 in multiparam_func.evolved_param_names:
            value = init_params[n2]
            if multiparam_func.param_dicts[n2].param_dict['param_type'] == 'float':
                params[n2] = float(value)
            elif multiparam_func.param_dicts[n2].param_dict['param_type'] == 'int': # pragma: no cover
                params[n2] = int(value)
            elif multiparam_func.param_dicts[n2].param_dict['param_type'] == 'bool':
                if isinstance(value, bool):
                    params[n2] = value
                elif isinstance(value, int):
                    params[n2] = bool(value)
                elif value.lower() in ('false','off','0'):
                    params[n2] = False
                elif value.lower() in ('true', 'on', '1'):
                    params[n2] = True
                else:
                    params[n2] = bool(value)
            else:
                raise RuntimeError(
                    "{0!s}: Uninterpretable EMPF {1!s} param_type {2!r} for {3!r}".format(
                        name, n2,
                        multiparam_func.param_dicts[n2].param_dict['param_type'],
                        multiparam_func))

        if return_partial:
            return _make_partial(multiparam_func.user_func, name=name, **params)

        instance = multiparam_func.init_instance(param_namespace=param_namespace)

        instance.set_values(**params)

        return instance

    def get_Evolved_MPF(self, # MORE THOROUGH TESTS NEEDED!
                        name, # type: str
                        which_type, # type: str
                        param_namespace=None
                        ):
        # type: (...) -> EvolvedMultiParameterFunction
        """Fetches a named EvolvedMultiParameterFunction instance."""

        # TODO: Accept in keyword format also (possibly use dict.update?);
        # probably package into function usable by get_func also
        if name in self.multiparam_func_dict[which_type]:
            mpfunc_dict = self.multiparam_func_dict[which_type] # type: Dict[str, MultiParameterFunction]
            return mpfunc_dict[name].init_instance(param_namespace=param_namespace)

        return self._get_func_inner(name, which_type,
                                    return_partial=False,
                                    param_namespace=param_namespace)

    def get_func(self, name, which_type, param_namespace=None):
        """
        Figures out what function, or function instance for multiparameter functions,
        is needed, and returns it.
        """
        if isinstance(name, EvolvedMultiParameterFunction) or hasattr(name, 'get_func'):
            return name.get_func()

        if name in self.norm_func_dict[which_type]:
            func_dict = self.norm_func_dict[which_type]
            return func_dict[name]

        return self._get_func_inner(name, which_type, return_partial=True,
                                    param_namespace=param_namespace)

    def add_func(self, name, user_func, which_type, **kwargs):
        """Adds a new activation/aggregation function, potentially multiparameter."""
        if not isinstance(user_func,
                          (types.BuiltinFunctionType,
                           types.FunctionType,
                           types.LambdaType)):
            raise InvalidFunctionError("A function object is required, not {0!s} ({1!s})".format(
                saferepr(user_func), name))

        if not hasattr(user_func, '__code__'):
            if kwargs:
                if isinstance(user_func, types.BuiltinFunctionType):
                    raise InvalidFunctionError(
                        "Cannot use built-in function {0!s} ({1!s}) ".format(
                            saferepr(user_func),name)
                        + "as multiparam {0!s} function - needs wrapping".format(which_type))
                else:
                    raise InvalidFunctionError(
                        "For a multiparam {0!s} function, need an object ".format(which_type)
                        + "with a __code__ attribute, not {1!s} ({2!s})".format(
                            saferepr(user_func), name))
            nfunc_dict = self.norm_func_dict[which_type]
            nfunc_dict[name] = user_func
            return

        func_code = user_func.__code__
        if func_code.co_argcount != (len(kwargs)+1):
            if func_code.co_argcount < (len(kwargs)+1):
                raise InvalidFunctionError(
                    "Function {0!s} ({1!s})".format(saferepr(user_func),name) +
                    " requires {0!s} args".format(func_code.co_argcount) +
                    " but was given {0!s} kwargs ({1!r})".format(
                        len(kwargs),kwargs))
            elif func_code.co_argcount > (len(kwargs)+1+len(self.shared_names[which_type])):
                raise InvalidFunctionError(
                    "Function {0!s} ({1!s})".format(saferepr(user_func),name) +
                    " requires {0:n} args".format(func_code.co_argcount) +
                    " but was given {0:n} kwargs ({1!r})".format(
                        len(kwargs),kwargs) +
                    " and have only {0:n} shared parameters".format(
                        len(self.shared_names[which_type])))

        if func_code.co_argcount == 1:
            self.norm_func_dict[which_type][name] = user_func
            return

        if ('(' in name) or (')' in name):
            raise InvalidFunctionError(
                "Invalid function name '{!s}' for {!s}".format(name,
                                                               saferepr(user_func))
                + " - multiparam function cannot have '(' or ')'")

        first_an = func_code.co_varnames[0]
        if first_an in kwargs:
            raise InvalidFunctionError(
                "First argument '{0!s}' of function {1!s}".format(first_an,
                                                                  saferepr(user_func))
                + " ({0!s}) may not be in kwargs {1!r}".format(name,kwargs))

        evolved_param_names = func_code.co_varnames[1:func_code.co_argcount]
        func_names_set = set(evolved_param_names)
        kwargs_names_set = set(kwargs.keys())
        shared_set = set(self.shared_names[which_type].keys())

        overlap = kwargs_names_set & shared_set
        if overlap:
            raise InvalidFunctionError(
                "Function {0!s} ({1!s}) kwargs {2!r} include ".format(saferepr(user_func),
                                                                name,list(kwargs.keys()))
                + "{0!r} also in shared parameters".format(overlap))

        missing2 = kwargs_names_set - func_names_set
        if missing2:
            raise InvalidFunctionError(
                "Function {0!s} ({1!s}) lacks arguments ".format(saferepr(user_func),
                                                                 name)
                + "{0!r} in kwargs {1!r}".format(missing2,kwargs))

        missing1 = func_names_set - kwargs_names_set
        if missing1:
            missing1 -= shared_set
        if missing1:
            raise InvalidFunctionError(
                "Function {0!s} ({1!s}) has arguments ".format(saferepr(user_func),
                                                               name)
                + "{0!r} not in shared parameters {1!r} or kwargs {2!r}".format(missing1,
                                                                                shared_set,
                                                                                kwargs))

##print("Adding function {0!r} ({1!s}) with kwargs {2!r} ({3!s})".format(user_func,name,
##                                                                  kwargs,type(kwargs)),
##              file=sys.stderr)

        func_dict = self.multiparam_func_dict[which_type]
        func_dict[name] = MultiParameterFunction(name=name,
                                                 which_type=which_type,
                                                 user_func=user_func,
                                                 evolved_param_names=evolved_param_names,
                                                 shared_names=self.shared_names[which_type],
                                                 **kwargs)

