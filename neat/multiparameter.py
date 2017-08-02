"""
Enables the use of activation and aggregation functions
with, as well as the usual input, one or more evolvable numeric parameters.
"""
##from __future__ import print_function

import functools
##import sys
import types

from neat.attributes import FloatAttribute
from neat.six_util import iteritems

class MultiParameterFunctionInstance(object):
    """
    Holds, initializes, and mutates the evolved parameters for one instance
    of a multiparameter function.
    """
    def __init__(self, name, multi_param_func):
        self.name = name
        self.multi_param_func = multi_param_func
        self.user_func = multi_param_func.user_func
        self.evolved_param_names = multi_param_func.evolved_param_names
        self.evolved_param_attributes = multi_param_func.evolved_param_attributes
        self.evolved_param_dicts = multi_param_func.evolved_param_dicts
        self.current_param_values = {}
        self.instance_name = ''

        self.init_value()

    def init_value(self, ignored_config=None):
        for n, m in iteritems(self.evolved_param_attributes):
            self.current_param_values[n] = m.init_value(self.multi_param_func)
        self.instance_name = self.name + '(' + ','.join([str(self.current_param_values[n])
                                                         for n in self.evolved_param_names]) + ')'

    def mutate_value(self, ignored_config=None):
        for n, m in iteritems(self.evolved_param_attributes):
            self.current_param_values[n] = m.mutate_value(self.current_param_values[n],
                                                          self.multi_param_func)
        self.instance_name = self.name + '(' + ','.join([str(self.current_param_values[n])
                                                         for n in self.evolved_param_names]) + ')'

    def __str__(self):
        return self.instance_name

    def distance(self, other):
        if not isinstance(other, MultiParameterFunctionInstance):
            return 1.0

        if self.name != other.name:
            return 1.0
        if self.instance_name == other.instance_name:
            return 0.0

        total_diff = 0.0
        for n in self.evolved_param_names:
            diff = abs(self.current_param_values[n] -
                       other.current_param_values[n])
            param_dict = self.evolved_param_dicts[n]
            total_diff += diff / max(1.0,abs(param_dict['max_value'] - param_dict['min_value']))
        return total_diff

    def copy(self):
        #print("{0!s}: Copying myself {1!r}".format(self.instance_name,self),file=sys.stderr)
        other = MultiParameterFunctionInstance(self.name, self.multi_param_func)
        for n in self.evolved_param_names:
            other.current_param_values[n] = self.current_param_values[n]
        other.instance_name = self.instance_name[:]
        return other

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, ignored_dict):
        return self.copy()

    def get_func(self):
        partial = functools.partial(self.user_func, **self.current_param_values)
        setattr(partial, '__name__', self.instance_name)
        if hasattr(self.user_func, '__doc__'):
            setattr(partial, '__doc__', self.user_func.__doc__)
        return partial

class MultiParameterFunction(object):
    """Holds and initializes configuration information for one multiparameter function."""
    def __init__(self, name, which_type, user_func, evolved_param_names, **evolved_param_dicts):
        self.name = name + "_function"
        self.orig_name = name
        self.which_type = which_type # activation or aggregation
        self.user_func = user_func
        self.evolved_param_names = evolved_param_names
        self.evolved_param_dicts = evolved_param_dicts
        self.evolved_param_attributes = {}

        for n in evolved_param_names:
            self.evolved_param_dicts[n].setdefault('init_type','uniform')
            param_dict = self.evolved_param_dicts[n]
            middle = (param_dict['max_value'] +
                      param_dict['min_value'])/2.0
            self.evolved_param_dicts[n].setdefault('init_mean', middle)
            for_stdev = min(abs(param_dict['max_value'] -
                                param_dict['init_mean']),
                            abs(param_dict['min_value'] -
                                param_dict['init_mean']))/2.0
            self.evolved_param_dicts[n].setdefault('init_stdev', for_stdev)
            # below here is mainly intended for users wanting to use built-in
            # multiparameter functions without too much initialization worries
            self.evolved_param_dicts[n].setdefault('replace_rate', 0.1)
            mutate_rate = min((1-param_dict['replace_rate']),(param_dict['replace_rate']*5.0))
            self.evolved_param_dicts[n].setdefault('mutate_rate', mutate_rate)
            # actual standard deviation of uniform distribution is width/sqrt(12) -
            # use of 1/4 range in the uniform distribution FloatAttribute setup
            # (and thus the above) is to make it easier to figure out how to
            # get a given initialization range that is not the same as the
            # overall min/max range.
            mutate_power = ((param_dict['replace_rate']/param_dict['mutate_rate'])*
                            (abs(param_dict['max_value']-param_dict['min_value'])/pow(12.0,0.5)))
            self.evolved_param_dicts[n].setdefault('mutate_power', mutate_power)
            
            tmp_name = "{0}_{1}".format(name,n)
            self.evolved_param_attributes[n] = FloatAttribute(name=tmp_name,
                                                              **param_dict)
            for x, y in iteritems(param_dict):
                setattr(self, self.evolved_param_attributes[n].config_item_name(x), y)

    def init_instance(self):
        return MultiParameterFunctionInstance(self.orig_name, self)

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
            raise InvalidFunctionError("Called with uncertain name '{!s}'".format(name))
        return False

    def is_multiparameter(self, name, which_type):
        if name.endswith(')'):
            raise InvalidFunctionError("Called with uncertain name '{!s}'".format(name))
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
            return mpfunc_dict[name] # Allows for altering configuration, although tricky re initialization + already-existing ones
        raise UnknownFunctionError("Unknown {!s} function {!r}".format(which_type,name))

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

        func_name = name[:(param_start-1)]
        if not func_name in self.multiparam_func_dict[which_type]:
            raise UnknownFunctionError("Unknown {0!s} function {1!r} (from {2!r})".
                                       format(which_type,func_name,name))
        multiparam_func = self.multiparam_func_dict[which_type][func_name]

        param_nums = map(float, name[(param_start+1):(len(name)-2)].split(','))

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

        if isinstance(user_func, types.BuiltinFunctionType): # TODO: Test!
            if kwargs:
                raise InvalidFunctionError(
                    "Cannot use built-in function {0!r} ({1!s}) as multiparam {2!s} function - needs wrapping".format(
                        user_func, name, which_type))
            nfunc_dict = self.norm_func_dict[which_type]
            nfunc_dict[name] = user_func
            return

        if not hasattr(user_func, '__code__'):
            raise InvalidFunctionError(
                "An object with __code__ attribute is required, not {0!r} ({1!s})".format(user_func,
                                                                                          name))
        
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

