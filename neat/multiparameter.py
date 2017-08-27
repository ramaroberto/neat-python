"""
Enables the use of activation and aggregation functions
with, as well as the usual input, one or more evolvable numeric parameters.
"""
from __future__ import division

import collections
import copy
import functools
##import sys
import types
import warnings

from math import sqrt
from pprint import saferepr

from neat.attributes import FloatAttribute, BoolAttribute
from neat.math_util import NORM_EPSILON
from neat.repr_util import repr_extract_function_name
from neat.six_util import iteritems, iterkeys

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

class EvolvedMultiParameterFunction(object):
    """
    Holds, initializes, and mutates the evolved parameters for one instance
    of a multiparameter function.
    """
    def __init__(self, name, multi_param_func):
        self.name = name
        assert isinstance(multi_param_func,
                          MultiParameterFunction), "MPF {0!s} type '{1!s}' ({2!s})".format(
            name, type(multi_param_func), saferepr(multi_param_func))
        self.multi_param_func = multi_param_func
        self.user_func = multi_param_func.user_func
        self.evolved_param_names = multi_param_func.evolved_param_names
        self.evolved_param_attributes = multi_param_func.evolved_param_attributes
        self.evolved_param_dicts = multi_param_func.evolved_param_dicts
        self.current_param_values = {}

        self.init_value()

    @property
    def instance_name(self):
        return self.name + '(' + ','.join([self.user_func.__code__.co_varnames[0]]
                                          + [str(self.current_param_values[n])
                                             for n in self.evolved_param_names]) + ')'

    def init_value(self, ignored_config=None):
        for n, m in iteritems(self.evolved_param_attributes):
            self.current_param_values[n] = m.init_value(self.multi_param_func)

    def mutate_value(self, ignored_config=None):
        for n, m in iteritems(self.evolved_param_attributes):
            self.current_param_values[n] = m.mutate_value(self.current_param_values[n],
                                                          self.multi_param_func)

    def set_values(self, **param_values): # TEST NEEDED!
        for n, val in iteritems(param_values):
            if n in self.evolved_param_names:
                self.current_param_values[n] = val
            else:
                raise LookupError(
                    "Parameter name {0!s} (val {1!s}) not among known ({2!s}) for {3!s}".format(
                        saferepr(n), saferepr(val), self.evolved_param_names, self.name))

    def get_values(self, n=None, do_copy=True):
        if n is not None:
            return self.current_param_values[n]
        if do_copy:
            return copy.deepcopy(self.current_param_values)
        return self.current_param_values

    def __str__(self):
        return self.instance_name

    def __repr__(self):
        return "EvolvedMultiParameterFunction({0!r}, {1!r}).set_values({2!r})".format(
            self.name, self.multi_param_func, self.current_param_values)

    def distance(self, other):
        """Determine distance between two activation or aggregation functions."""
        if not isinstance(other, EvolvedMultiParameterFunction):
            return 1.0

        if self.name != other.name:
            return 1.0
        if self.instance_name == other.instance_name:
            return 0.0

        max_diff = 0.0
        for n in self.evolved_param_names:
            param_dict = self.evolved_param_dicts[n]
            if param_dict['param_type'] in ('float', 'int'):
                diff = abs(self.current_param_values[n] -
                           other.current_param_values[n])
                if diff:
                    div_by = max(NORM_EPSILON,
                                 abs(param_dict['max_value'] - param_dict['min_value']))
                    this_diff = diff / div_by
                    if this_diff > 1.0: # pragma: no cover
                        raise RuntimeError(
                            "This_diff {0:n} over 1.0 (diff {1:n}, div_by {2:n})".format(
                                this_diff, diff, div_by))
                    max_diff = max(max_diff, this_diff)
            elif param_dict['param_type'] == 'bool':
                if self.current_param_values[n] != other.current_param_values[n]:
                    return 1.0
            else:
                raise ValueError("Unknown what to do with param_type {0!s} for {1!s}".format(
                    saferepr(param_dict['param_type']), self.name))
        return max_diff

    def copy(self):
        return copy.copy(self)

    def __copy__(self):
        other = EvolvedMultiParameterFunction(self.name[:], copy.copy(self.multi_param_func))
        other.current_param_values = copy.deepcopy(self.current_param_values)
        return other

    def __deepcopy__(self, memo_dict):
        other = EvolvedMultiParameterFunction(self.name[:],
                                              copy.deepcopy(self.multi_param_func,memo_dict))
        other.current_param_values = copy.deepcopy(self.current_param_values,memo_dict)
        return other

    def get_func(self):
        return _make_partial(self.user_func, self.instance_name, **self.current_param_values)

class MultiParameterFunction(object):
    """Holds and initializes configuration information for one multiparameter function."""
    def __init__(self,
                 name,
                 which_type,
                 user_func,
                 evolved_param_names,
                 full_init_defaults=True,
                 **evolved_param_dicts):
        self.name = name + "_MPF"
        self.orig_name = name
        self.which_type = which_type # activation or aggregation
        self.user_func = user_func
        self.evolved_param_names = evolved_param_names # for the ordering
        self.evolved_param_dicts = evolved_param_dicts
        self.evolved_param_attributes = {}

        for n in evolved_param_names:
            self.init_defaults(n, full=full_init_defaults)

    def init_defaults(self, n, full=True):
        """
        Initializes (or re-initializes after user settings changes, provided the user deletes
        any old settings not altered) defaults for one parameter's attribute settings
        for a multiparameter function.
        """
        self.evolved_param_dicts[n].setdefault('param_type', 'float')
        param_dict = self.evolved_param_dicts[n]
        tmp_name = "{0}_{1}".format(self.name,n)

        if param_dict['param_type'] in ('float','int'):
            if full:
                self.evolved_param_dicts[n].setdefault('init_type','uniform')

                self.evolved_param_dicts[n].setdefault('max_init_value',
                                                       param_dict['max_value'])
                self.evolved_param_dicts[n].setdefault('min_init_value',
                                                       param_dict['min_value'])
                self.evolved_param_dicts[n].setdefault('max_value',
                                                       param_dict['max_init_value'])
                self.evolved_param_dicts[n].setdefault('min_value',
                                                       param_dict['min_init_value'])

                middle = (param_dict['max_init_value'] +
                          param_dict['min_init_value'])/2.0
                self.evolved_param_dicts[n].setdefault('init_mean', middle)
                # below here is mainly intended for users wanting to use built-in
                # multiparameter functions without too much initialization worries
                self.evolved_param_dicts[n].setdefault('replace_rate', 0.1)
                mutate_rate = min((1.0-param_dict['replace_rate']),
                                  (param_dict['replace_rate']*5.0))
                self.evolved_param_dicts[n].setdefault('mutate_rate', mutate_rate)
                for_stdev = min(abs(param_dict['max_init_value'] -
                                    param_dict['init_mean']),
                                abs(param_dict['min_init_value'] -
                                    param_dict['init_mean']))/2.0
                if param_dict['init_type'] == 'uniform':
                    self.evolved_param_dicts[n].setdefault('init_stdev', for_stdev)
                    # actual standard deviation of uniform distribution is width/sqrt(12) -
                    # use of 1/4 range in the uniform distribution FloatAttribute setup
                    # (and thus the above) is to make it easier to figure out how to
                    # get a given initialization range that is not the same as the
                    # overall min/max range.
                else:
                    self.evolved_param_dicts[n].setdefault('init_stdev',
                                                           ((4.0*for_stdev)/sqrt(12.0)))
                if param_dict['mutate_rate'] > 0:
                    mutate_power = (min(1.0,(param_dict['replace_rate']/
                                             param_dict['mutate_rate']))*
                                    (abs(param_dict['max_init_value']
                                         -param_dict['min_init_value'])/
                                     sqrt(12.0)))
                    self.evolved_param_dicts[n].setdefault('mutate_power', mutate_power)
                else: # pragma: no cover
                    self.evolved_param_dicts[n].setdefault('mutate_power', 0.0)
                    warnings.warn("Mutate_rate for param {0!r} of {1!r} is {2:n}".format(
                        n, self.orig_name, param_dict['mutate_rate']),
                                  RuntimeWarning)

            param_dict2 = copy.copy(param_dict)
            del param_dict2['param_type']
            del param_dict2['min_init_value']
            del param_dict2['max_init_value']

            self.evolved_param_attributes[n] = FloatAttribute(name=tmp_name, # TODO: IntAttribute
                                                              default_ok=True,
                                                              **param_dict2)
        elif param_dict['param_type'] == 'bool':
            if full:
                max_rate_add = max(param_dict.get('rate_to_true_add',0.0),
                                   param_dict.get('rate_to_false_add',0.0))
                self.evolved_param_dicts[n].setdefault('mutate_rate',
                                                       min(0.1,max(0.0,(1.0-max_rate_add))))
                self.evolved_param_dicts[n].setdefault('default', 'random')
                self.evolved_param_dicts[n].setdefault('rate_to_true_add', 0.0)
                self.evolved_param_dicts[n].setdefault('rate_to_false_add', 0.0)

            param_dict2 = copy.copy(param_dict)
            del param_dict2['param_type']

            self.evolved_param_attributes[n] = BoolAttribute(name=tmp_name,
                                                             default_ok=True,
                                                             **param_dict2)
        else:
            raise ValueError(
                "Unknown param_type {0!r} for MultiParameterFunction {1!s}".format(
                    param_dict['param_type'], self.orig_name))

        for x, y in iteritems(self.evolved_param_dicts[n]):
            # so that this can be used as a config for the attribute
            setattr(self, self.evolved_param_attributes[n].config_item_name(x), y)

    def init_instance(self):
        return EvolvedMultiParameterFunction(self.orig_name, self)

    def copy_and_change(self,
                        del_not_changed=False,
                        del_param_dicts=None,
                        new_param_dicts=None):
        """
        Makes a copy of the MultiParameterFunction instance, does deletions and
        substitutions, initializes any remaining defaults, and returns the new instance.
        If del_not_changed is True, any values for parameters in new_param_dicts that are not
        set will be deleted and reinitialized.
        """
        new = copy.deepcopy(self)
        n_done = set([])
        if del_param_dicts is not None:
            for n in iterkeys(del_param_dicts):
                if isinstance(del_param_dicts[n], (collections.Sequence,collections.Iterable)):
                    for x in del_param_dicts[n]:
                        del new.evolved_param_dicts[n][x]
                elif del_param_dicts[n] is None:
                    param_type = new.evolved_param_dicts[n]['param_type']
                    del new.evolved_param_dicts[n]
                    new.evolved_param_dicts[n] = {'param_type':param_type}
                elif isinstance(del_param_dicts[n], collections.Hashable):
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
                if del_not_changed:
                    param_type = new.evolved_param_dicts[n]['param_type']
                    del new.evolved_param_dicts[n]
                    new.evolved_param_dicts[n] = {'param_type':param_type}
                for x, y in iteritems(new_param_dicts[n]):
                    new.evolved_param_dicts[n][x] = y
                n_done.add(n)
        for n in n_done:
            new.init_defaults(n)
        return new

    def __repr__(self): # TEST NEEDED! Should be able to duplicate by using this as an init...
        to_return_list = []
        to_return_list.append('orig_name=' + repr(self.orig_name))
        to_return_list.append('which_type=' + repr(self.which_type))
        to_return_list.append('user_func=' + repr_extract_function_name(self.user_func,
                                                                        with_module=True))
        to_return_list.append('evolved_param_names=' + repr(self.evolved_param_names))
        for n in self.evolved_param_names:
            to_return_list.append(repr(n) + '=' + repr(self.evolved_param_dicts[n]))
        return 'MultiParameterFunction(' + ",".join(to_return_list) + ')'

    def __copy__(self):
        new_evolved_param_dicts = {}
        for n in self.evolved_param_names:
            new_evolved_param_dicts[n] = copy.copy(self.evolved_param_dicts[n])
        return MultiParameterFunction(name=self.orig_name[:], which_type=self.which_type[:],
                                      user_func=copy.copy(self.user_func),
                                      evolved_param_names=self.evolved_param_names[:],
                                      full_init_defaults=False,
                                      **new_evolved_param_dicts)

    def __deepcopy__(self, memo_dict):
        return MultiParameterFunction(name=self.orig_name[:], which_type=self.which_type[:],
                                      user_func=copy.deepcopy(self.user_func, memo_dict),
                                      evolved_param_names=copy.deepcopy(self.evolved_param_names,
                                                                        memo_dict),
                                      full_init_defaults=False,
                                      **copy.deepcopy(self.evolved_param_dicts, memo_dict))


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
        return bool(name in self.multiparam_func_dict[which_type])

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
        """Fetches the named MultiParameterFunction instance."""

        if name in self.multiparam_func_dict[which_type]:
            mpfunc_dict = self.multiparam_func_dict[which_type] # type: Dict[str, MultiParameterFunction]
            # Allows for altering configuration - preferably via copy_and_change!
            return mpfunc_dict[name]
        raise UnknownFunctionError("Unknown {!s} MPF function {!r}".format(which_type,name))

    def _get_func_inner(self,
                       name, # type: str
                       which_type, # type: str
                       return_partial # type: bool
                       ):
        if not name.endswith(')'):
            raise UnknownFunctionError("Unknown {!s} function {!r} - no end )".
                                       format(which_type,name))

        param_start = name.find('(')
        if param_start < 0:
            raise UnknownFunctionError("Unknown {!s} function {!r} - no start (".
                                       format(which_type,name))

        func_name = name[:param_start]
        if func_name not in self.multiparam_func_dict[which_type]:
            raise UnknownFunctionError("Unknown {0!s} function {1!r} (from {2!r})".
                                       format(which_type,func_name,name))
        multiparam_func = self.multiparam_func_dict[which_type][func_name]

        param_values = name[(param_start+1):(len(name)-1)].split(',')

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
            if multiparam_func.evolved_param_dicts[n2]['param_type'] == 'float':
                params[n2] = float(value)
            elif multiparam_func.evolved_param_dicts[n2]['param_type'] == 'int': # pragma: no cover
                params[n2] = int(value)
            elif multiparam_func.evolved_param_dicts[n2]['param_type'] == 'bool':
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
                        multiparam_func.evolved_param_dicts[n2]['param_type'],
                        multiparam_func))

        if return_partial:
            return _make_partial(multiparam_func.user_func, name=name, **params)

        instance = multiparam_func.init_instance()

        instance.set_values(**params)

        return instance

    def get_Evolved_MPF(self, # MORE THOROUGH TESTS NEEDED!
                         name, # type: str
                         which_type # type: str
                        ):
        # type: (...) -> EvolvedMultiParameterFunction
        """Fetches a named EvolvedMultiParameterFunction instance."""

        # TODO: Accept in keyword format also (possibly use dict.update?);
        # probably package into function usable by get_func also
        if name in self.multiparam_func_dict[which_type]:
            mpfunc_dict = self.multiparam_func_dict[which_type] # type: Dict[str, MultiParameterFunction]
            return mpfunc_dict[name].init_instance()

        return self._get_func_inner(name, which_type, return_partial=False)

    def get_func(self, name, which_type):
        """
        Figures out what function, or function instance for multiparameter functions,
        is needed, and returns it.
        """
        if isinstance(name, EvolvedMultiParameterFunction) or hasattr(name, 'get_func'):
            return name.get_func()

        if name in self.norm_func_dict[which_type]:
            func_dict = self.norm_func_dict[which_type]
            return func_dict[name]

        return self._get_func_inner(name, which_type, return_partial=True)

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
            raise InvalidFunctionError("Function {0!s} ({1!s})".format(saferepr(user_func),name) +
                                       " requires {0!s} args".format(func_code.co_argcount) +
                                       " but was given {0!s} kwargs ({1!r})".format(len(kwargs),
                                                                                    kwargs))

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

        missing1 = func_names_set - kwargs_names_set
        if missing1:
            raise InvalidFunctionError(
                "Function {0!s} ({1!s}) has arguments ".format(saferepr(user_func),
                                                               name)
                + "{0!r} not in kwargs {1!r}".format(missing1,
                                                     kwargs_names_set))
        missing2 = kwargs_names_set - func_names_set
        if missing2:
            raise InvalidFunctionError(
                "Function {0!s} ({1!s}) lacks arguments ".format(saferepr(user_func),
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

