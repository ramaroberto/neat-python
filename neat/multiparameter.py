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

from math import sqrt, isnan

from neat.attributes import FloatAttribute, BoolAttribute
from neat.math_util import NORM_EPSILON
from neat.six_util import iteritems

class MultiParameterFunctionInstance(object):
    """
    Holds, initializes, and mutates the evolved parameters for one instance
    of a multiparameter function.
    """
    def __init__(self, name, multi_param_func):
        self.name = name
        assert isinstance(multi_param_func, MultiParameterFunction), "multi_param_func {0!s} type '{1!s}' ({2!r}) bad".format(
            name, type(multi_param_func), multi_param_func)
        self.multi_param_func = multi_param_func
        self.user_func = multi_param_func.user_func
        self.evolved_param_names = multi_param_func.evolved_param_names
        self.evolved_param_attributes = multi_param_func.evolved_param_attributes
        self.evolved_param_dicts = multi_param_func.evolved_param_dicts
        self.current_param_values = {}
        self.instance_name = ''

        self.init_value()

    def __set_instance_name(self):
        self.instance_name = self.name + '(' + ','.join([str(self.current_param_values[n])
                                                         for n in self.evolved_param_names]) + ')'

    def init_value(self, ignored_config=None):
        for n, m in iteritems(self.evolved_param_attributes):
            self.current_param_values[n] = m.init_value(self.multi_param_func)
        self.__set_instance_name()

    def mutate_value(self, ignored_config=None):
        for n, m in iteritems(self.evolved_param_attributes):
            self.current_param_values[n] = m.mutate_value(self.current_param_values[n],
                                                          self.multi_param_func)
        self.__set_instance_name()

    def set_values(self, **param_values): # TEST NEEDED; Perhaps do as @property?
        for n, val in iteritems(param_values):
            if n in self.evolved_param_names:
                if not isnan(val):
                    self.current_param_values[n] = val
            else:
                raise LookupError(
                    "Parameter name {0!r} (val {1!r}) not among known ({2!s}) for {3!s}".format(
                        n, val, self.evolved_param_names, self.name))
        self.__set_instance_name()

    def __str__(self):
        return self.instance_name

    def distance(self, other):
        if not isinstance(other, MultiParameterFunctionInstance):
            return 1.0

        if self.name != other.name:
            return 1.0
        if self.instance_name == other.instance_name:
            return 0.0

        max_diff = 0.0
        for n in self.evolved_param_names:
            diff = abs(self.current_param_values[n] -
                       other.current_param_values[n])
            if diff:
                param_dict = self.evolved_param_dicts[n]
                this_diff = diff / max(NORM_EPSILON,abs(param_dict['max_value'] - param_dict['min_value']))
                max_diff = max(max_diff, this_diff)
        return max_diff

    def copy(self):
        #print("{0!s}: Copying myself {1!r}".format(self.instance_name,self),file=sys.stderr)
        other = MultiParameterFunctionInstance(self.name, self.multi_param_func.copy())
        for n in self.evolved_param_names:
            other.current_param_values[n] = self.current_param_values[n]
        other.instance_name = self.instance_name[:]
        return other

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo_dict):
        other = MultiParameterFunctionInstance(self.name[:], self.multi_param_func.deepcopy(memo_dict))
        for n in self.evolved_param_names:
            other.current_param_values[n] = self.current_param_values[n]
        other.instance_name = self.instance_name[:]
        return other

    def get_func(self):
        partial = functools.partial(self.user_func, **self.current_param_values)
        setattr(partial, '__name__', self.instance_name)
        if hasattr(self.user_func, '__doc__'):
            setattr(partial, '__doc__', self.user_func.__doc__)
        return partial

class MultiParameterFunction(object):
    """Holds and initializes configuration information for one multiparameter function."""
    def __init__(self, name, which_type, user_func, evolved_param_names, **evolved_param_dicts):
        self.name = name + "_MPF"
        self.orig_name = name
        self.which_type = which_type # activation or aggregation
        self.user_func = user_func
        self.evolved_param_names = evolved_param_names
        self.evolved_param_dicts = evolved_param_dicts
        self.evolved_param_attributes = {}

        for n in evolved_param_names:
            self.init_defaults(n)

    def init_defaults(self, n):
        """
        Initializes (or re-initializes after user settings changes - the user will need to
        remove existing entries in the evolved_param_dicts[n] dictionary, and keep in mind
        that MultiParameterFunctionInstances generally share MultiParameterFunction instances)
        defaults for one parameter's FloatAttribute settings for a multiparameter function.
        """
        self.evolved_param_dicts[n].setdefault('param_type', 'float')
        param_dict = self.evolved_param_dicts[n]
        tmp_name = "{0}_{1}".format(self.name,n)
        
        if param_dict['param_type'] in ('float','int'):
            self.evolved_param_dicts[n].setdefault('init_type','uniform')
 
            middle = (param_dict['max_value'] +
                      param_dict['min_value'])/2.0
            self.evolved_param_dicts[n].setdefault('init_mean', middle)
            # below here is mainly intended for users wanting to use built-in
            # multiparameter functions without too much initialization worries
            self.evolved_param_dicts[n].setdefault('replace_rate', 0.1)
            mutate_rate = min((1.0-param_dict['replace_rate']),(param_dict['replace_rate']*5.0))
            self.evolved_param_dicts[n].setdefault('mutate_rate', mutate_rate)
            for_stdev = min(abs(param_dict['max_value'] -
                                param_dict['init_mean']),
                            abs(param_dict['min_value'] -
                                param_dict['init_mean']))/2.0
            if param_dict['init_type'] == 'uniform':
                self.evolved_param_dicts[n].setdefault('init_stdev', for_stdev)
                # actual standard deviation of uniform distribution is width/sqrt(12) -
                # use of 1/4 range in the uniform distribution FloatAttribute setup
                # (and thus the above) is to make it easier to figure out how to
                # get a given initialization range that is not the same as the
                # overall min/max range.
            else:
                self.evolved_param_dicts[n].setdefault('init_stdev', ((4.0*for_stdev)/sqrt(12.0)))
            if param_dict['mutate_rate'] > 0:
                mutate_power = (min(1.0,(param_dict['replace_rate']/param_dict['mutate_rate']))*
                                (abs(param_dict['max_value']-param_dict['min_value'])/sqrt(12.0)))
                self.evolved_param_dicts[n].setdefault('mutate_power', mutate_power)
            else:
                self.evolved_param_dicts[n].setdefault('mutate_power', 0.0)

            param_dict2 = copy.deepcopy(param_dict)
            del param_dict2['param_type']

            self.evolved_param_attributes[n] = FloatAttribute(name=tmp_name, # TODO: IntAttribute
                                                              **param_dict2)
        elif param_dict['param_type'] == 'bool':
            self.evolved_param_dicts[n].setdefault('mutate_rate', 0.1)

            param_dict2 = copy.deepcopy(param_dict)
            del param_dict2['param_type']

            self.evolved_param_attributes[n] = BoolAttribute(name=tmp_name,
                                                             **param_dict2)
        else:
            raise RuntimeError(
                "Unknown param_type {0!r} for MultiParameterFunction {1!s}".format(
                    param_dict['param_type'], self.orig_name))

        for x, y in iteritems(param_dict): # so that this can be used as a config for the attribute
            setattr(self, self.evolved_param_attributes[n].config_item_name(x), y)

    def init_instance(self):
        return MultiParameterFunctionInstance(self.orig_name, self)

##    def set_attribute_settings(self, n, **param_dict): # TEST NEEDED! Perhaps do as @property?
##        """
##        Sets the FloatAttribute settings for one of a function's parameters. Note that this will
##        change these for all MultiParameterFunctionInstances derived from MultiParameterFunction
##        class instance, although such changes will not have an effect until the next mutation
##        (including reinitialization). It is advisable to make a copy first, and alter that.
##        """
##        for x, y in iteritems(param_dict):
##            self.evolved_param_dicts[n][x] = y
##        self.init_defaults(n)

    def __repr__(self): # TEST NEEDED! Should be able to duplicate by using this as an init...
        to_return_list = [self.orig_name,
                          self.which_type,
                          self.user_func,
                          self.evolved_param_names]
        to_return_list = list(map(repr,to_return_list))
        for n in self.evolved_param_names:
            to_return_list.append(n + '=' + repr(self.evolved_param_dicts[n]))
        return str(self.__class__) + '(' + ",".join(to_return_list) + ')'

    def copy(self):
        return MultiParameterFunction(self.orig_name, self.which_type, self.user_func,
                                      self.evolved_param_names, **self.evolved_param_dicts)

    def deepcopy(self, memo_dict):
        return MultiParameterFunction(self.orig_name[:], self.which_type[:],
                                      copy.deepcopy(self.user_func, memo_dict),
                                      copy.deepcopy(self.evolved_param_names, memo_dict),
                                      **copy.deepcopy(self.evolved_param_dicts, memo_dict))

    def __copy__(self): # TEST NEEDED!
        return self.copy()

    def __deepcopy__(self, memo_dict): # TEST NEEDED!
        return self.deepcopy(memo_dict)


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
        for which_type in list(which_types):
            self.norm_func_dict[which_type] = {}
            self.multiparam_func_dict[which_type] = {}

    def is_valid_func(self, name, which_type):
        if name in self.multiparam_func_dict[which_type]:
            return True
        if name in self.norm_func_dict[which_type]:
            return True
        if name.endswith(')'):
            raise UnknownFunctionError("Called with uncertain name '{!s}'".format(name))
        return False

    def is_multiparameter(self, name, which_type):
        if name.endswith(')'):
            raise UnknownFunctionError("Called with uncertain name '{!s}'".format(name))
        return name in self.multiparam_func_dict[which_type]

    def init_multiparameter(self, name, instance, ignored_config=None):
        which_type = instance.name
        multiparam_func_dict = self.multiparam_func_dict[which_type]
        multiparam_func = multiparam_func_dict[name]
        return multiparam_func.init_instance()

    def get_MPF(self,
                name, # type: str
                which_type # type: str
                ):
        # type: (...) -> MultiParameterFunction

        if name in self.multiparam_func_dict[which_type]:
            mpfunc_dict = self.multiparam_func_dict[which_type] # type: Dict[str, MultiParameterFunction]
            # Allows for altering configuration, although tricky re already-existing ones
            return mpfunc_dict[name]
        raise UnknownFunctionError("Unknown {!s} MPF function {!r}".format(which_type,name))

    def get_MPF_Instance(self, # MORE THOROUGH TESTS NEEDED!
                         name, # type: str
                         which_type # type: str
                        ):
        # type: (...) -> MultiParameterFunctionInstance
        # TODO: Accept in keyword format also; probably package into function usable by get_func also
        if name in self.multiparam_func_dict[which_type]:
            mpfunc_dict = self.multiparam_func_dict[which_type] # type: Dict[str, MultiParameterFunction]
            return mpfunc_dict[name].init_instance()

        if not name.endswith(')'):
            raise UnknownFunctionError("Unknown {!s} MPF function {!r} - no end ')'".format(
                which_type,name))

        param_start = name.find('(')
        if param_start < 0:
            raise UnknownFunctionError("Unknown {!s} MPF function {!r} - no start '('".
                                       format(which_type,name))

        func_name = name[:param_start]
        if not func_name in self.multiparam_func_dict[which_type]:
            raise UnknownFunctionError("Unknown {0!s} MPF function {1!r} (from {2!r})".
                                       format(which_type,func_name,name))
        multiparam_func = self.multiparam_func_dict[which_type][func_name]

        param_nums = list(map(float, name[(param_start+1):(len(name)-1)].split(',')))

        if len(multiparam_func.evolved_param_names) < len(param_nums):
            raise RuntimeError(
                "Too many ({0:n}) param_nums in name {1!r} - should be max {2:n}".format(
                    len(param_nums), name, len(multiparam_func.evolved_param_names)))
        elif len(multiparam_func.evolved_param_names) > len(param_nums):
            warnings.warn(
                "MPFInstance name {0!r} has only {1:n} param_nums, while function takes {2:n}".format(
                    name, len(param_nums), len(multiparam_func.evolved_param_names)))

        params = dict(zip(multiparam_func.evolved_param_names, param_nums))

        instance = multiparam_func.init_instance()

        instance.set_values(**params)

        return instance
        

    def get_func(self, name, which_type):
        """
        Figures out what function, or function instance for multiparameter functions,
        is needed, and returns it.
        """
        if isinstance(name, MultiParameterFunctionInstance):
            return name.get_func()
        if hasattr(name, 'get_func'):
            return name.get_func()

        if name in self.norm_func_dict[which_type]:
            func_dict = self.norm_func_dict[which_type]
            return func_dict[name]

        if not name.endswith(')'):
            raise UnknownFunctionError("Unknown {!s} function {!r} - no end )".
                                       format(which_type,name))

        param_start = name.find('(')
        if param_start < 0:
            raise UnknownFunctionError("Unknown {!s} function {!r} - no start (".
                                       format(which_type,name))

        func_name = name[:param_start]
        if not func_name in self.multiparam_func_dict[which_type]:
            raise UnknownFunctionError("Unknown {0!s} function {1!r} (from {2!r})".
                                       format(which_type,func_name,name))
        multiparam_func = self.multiparam_func_dict[which_type][func_name]

        param_nums = list(map(float, name[(param_start+1):(len(name)-1)].split(',')))

        params = dict(zip(multiparam_func.evolved_param_names, param_nums))

        partial = functools.partial(multiparam_func.user_func, **params)
        setattr(partial, '__name__', name)
        if hasattr(multiparam_func.user_func, '__doc__'):
            setattr(partial, '__doc__', multiparam_func.user_func.__doc__)
        return partial

    def add_func(self, name, user_func, which_type, **kwargs):
        """Adds a new activation/aggregation function, potentially multiparameter."""
        if not isinstance(user_func,
                          (types.BuiltinFunctionType,
                           types.FunctionType,
                           types.LambdaType)):
            raise InvalidFunctionError("A function object is required, not {0!r} ({1!s})".format(
                user_func, name))

        if not hasattr(user_func, '__code__'):
            if kwargs:
                if isinstance(user_func, types.BuiltinFunctionType):
                    raise InvalidFunctionError(
                        "Cannot use built-in function {0!r} ({1!s}) ".format(user_func,name)
                        + "as multiparam {0!s} function - needs wrapping".format(which_type))
                else:
                    raise InvalidFunctionError(
                        "For a multiparam {0!s} function, need an object ".format(which_type)
                        + "with a __code__ attribute, not {1!r} ({2!s})".format(
                            user_func, name))
            nfunc_dict = self.norm_func_dict[which_type]
            nfunc_dict[name] = user_func
            return
        
        func_code = user_func.__code__
        if func_code.co_argcount != (len(kwargs)+1):
            raise InvalidFunctionError("Function {0!r} ({1!s})".format(user_func,name) +
                                       " requires {0!s} args".format(func_code.co_argcount) +
                                       " but was given {0!s} kwargs ({1!r})".format(len(kwargs),
                                                                                    kwargs))

        if func_code.co_argcount == 1:
            func_dict = self.norm_func_dict[which_type]
            func_dict[name] = user_func
            return

        if ('(' in name) or (')' in name):
            raise InvalidFunctionError("Invalid function name '{!s}' for {!r}".format(name,
                                                                                      user_func)
                                       + " - multiparam function cannot have '(' or ')'")

        first_an = func_code.co_varnames[0]
        if first_an in kwargs:
            raise InvalidFunctionError("First argument '{0!s}' of function {1!r}".format(first_an,
                                                                                         user_func)
                                       + " ({0!s}) may not be in kwargs {1!r}".format(name,kwargs))

        evolved_param_names = func_code.co_varnames[1:func_code.co_argcount]
        func_names_set = set(evolved_param_names)
        kwargs_names_set = set(kwargs.keys())
        
        missing1 = func_names_set - kwargs_names_set
        if missing1:
            raise InvalidFunctionError("Function {0!r} ({1!s}) has arguments ".format(user_func,
                                                                                       name)
                                       + "{0!r} not in kwargs {1!r}".format(missing1,
                                                                             kwargs_names_set))
        missing2 = kwargs_names_set - func_names_set
        if missing2:
            raise InvalidFunctionError("Function {0!r} ({1!s}) lacks arguments ".format(user_func,
                                                                                         name)
                                       + "{0!r} in kwargs {1!r}".format(missing2,kwargs))

##print("Adding function {0!r} ({1!s}) with kwargs {2!r} ({3!s})".format(user_func,name,
##                                                                  kwargs,type(kwargs)),
##              file=sys.stderr)

        func_dict = self.multiparam_func_dict[which_type]
        func_dict[name] = MultiParameterFunction(name=name, which_type=which_type,
                                                 user_func=user_func,
                                                 evolved_param_names=evolved_param_names,
                                                 **kwargs)

