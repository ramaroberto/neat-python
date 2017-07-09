"""
Enables the use of activation and aggregation functions
with multiple evolvable numeric parameters.
"""
import types
import weakref

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
        self.instance_name = self.name + '(' + ','.join([self.current_param_values[n]
                                                         for n in self.evolved_param_names]) + ')'

    def mutate_value(self, ignored_config=None):
        for n, m in iteritems(self.evolved_param_attributes):
            self.current_param_values[n] = m.mutate_value(self.current_param_values[n],
                                                          self.multi_param_func)
        self.instance_name = self.name + '(' + ','.join([self.current_param_values[n]
                                                         for n in self.evolved_param_names]) + ')'

    def __str__(self):
        return self.instance_name

    def distance(self, other):
        if not isinstance(other, MultiParameterFunctionInstance):
            return 1

        if self.name != other.name:
            return 1

        total_diff = 0
        for n in self.evolved_param_names:
            diff = abs(self.current_param_values[n] -
                       other.current_param_values[n])
            param_dict = self.evolved_param_dicts[n]
            total_diff += diff / abs(param_dict['max_value'] -
                                     param_dict['min_value'])
        return total_diff

    def copy(self):
        other = MultiParameterFunctionInstance(self.name, self.multi_param_func)
        for n in self.evolved_param_names:
            other.current_param_values[n] = self.current_param_values[n]
        other.instance_name = self.instance_name
        return other
        

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
                      param_dict['min_value'])/2
            self.evolved_param_dicts[n].setdefault('init_mean', middle)
            for_stdev = min(abs(param_dict['max_value'] -
                                param_dict['init_mean']),
                            abs(param_dict['min_value'] -
                                param_dict['init_mean']))/2
            self.evolved_param_dicts[n].setdefault('init_stdev', for_stdev)
            
            tmp_name = "{0}_{1}".format(name,n)
            self.evolved_param_attributes[n] = FloatAttribute(name=tmp_name,
                                                              default_dict=param_dict)
            for x, y in iteritems(param_dict):
                setattr(self, self.evolved_param_attributes[n].config_item_name(x), y)

    def init_instance(self):
        return MultiParameterFunctionInstance(self.orig_name, self)

class InvalidFunction(TypeError):
    pass

def user_func_maker(user_func, *params):
    def func_instance(x):
        return user_func(x, params)

    return func_instance

class MultiParameterSet(object):
    """
    Holds the set of (potentially multiparameter) functions
    and contains methods for dealing with them.
    """
    def __init__(self, *which_types):
        for which_type in which_types:
            self.weak_dict[which_type] = weakref.WeakKeyDictionary()
            self.norm_func_dict[which_type] = {}
            self.multiparam_func_dict[which_type] = {}

    def is_valid_func(self, name, which_type):
        if name in self.multiparam_func_dict[which_type]:
            return True
        if name in self.norm_func_dict[which_type]:
            return True
        if (name.index('(') > -1) or (name.index(')') > -1) or (name.index(',') > -1):
            raise InvalidFunction("Called with uncertain name '{!s}'".format(name))
        return False

    def is_multiparameter(self, name, which_type):
        return name in self.multiparam_func_dict[which_type]

    def init_multiparameter(self, name, instance, ignored_config):
        which_type = instance.name
        multiparam_func_dict = self.multiparam_func_dict[which_type]
        multiparam_func = multiparam_func_dict[name]
        return multiparam_func.init_instance()

    def get_func(self, name, which_type):
        """
        Figures out what function, or function instance for multiparameter functions,
        is needed, and returns it.
        """
        if name in self.norm_func_dict[which_type]:
            func_dict = self.norm_func_dict[which_type]
            return func_dict[name]

        if hasattr(name, '__class__') and (name.__class__ == 'MultiParameterFunctionInstance'):
            weak_dict = self.weak_dict[which_type]
            try:
                return weak_dict[name.instance_name]
            except LookupError:
                weak_dict[name.instance_name] = user_func_maker(name.user_func,
                                                                [name.current_param_values[n]
                                                                 for n in
                                                                 name.evolved_param_names])
                return weak_dict[name.instance_name]

        if name in self.multiparam_func_dict[which_type]:
            func_dict = self.multiparam_func_dict[which_type]
            return func_dict[name] # XXX

        if name[-1] != ')':
            raise LookupError("Unknown function {!r} - no end )".
                              format(name))

        param_start = name.index('(')
        if param_start < 0:
            raise LookupError("Unknown function {!r} - no start (".
                              format(name))

        func_name = name[:(param_start-1)]
        if not func_name in self.multiparam_func_dict[which_type]:
            raise LookupError("Unknown function {!r} (from {!r})".
                              format(func_name,name))

        try:
            weak_dict = self.weak_dict[which_type]
            return weak_dict[name]
        except LookupError:
            func_dict = self.multiparam_func_dict[which_type]
            multiparam_func = func_dict[func_name]
            params = name[(param_start+1):(len(name)-2)].split(',')
            self.weak_dict[name] = user_func_maker(multiparam_func.user_func, params)
            return self.weak_dict[name]

    def add_func(self, name, user_func, which_type, **kwargs):
        """Adds a new activation/aggregation function, potentially multiparameter."""
        if not isinstance(user_func,
                          (types.BuiltinFunctionType,
                           types.FunctionType,
                           types.LambdaType)):
            raise InvalidFunction("A function object is required.")
        
        func_code = user_func.__code__
        if func_code.co_argcount != (len(kwargs)+1):
            raise InvalidFunction("Function {0!r} ({1!s})".format(user_func,name) +
                                  " requires {0!s} args".format(func_code.co_argcount) +
                                  " but was given {0!s} kwargs ({1!r})".format(len(kwargs),
                                                                               kwargs))

        if (name.index('(') > -1) or (name.index(')') > -1) or (name.index(',') > -1):
            raise InvalidFunction("Invalid function name '{!s}' for {!r}".format(name,
                                                                                 user_func)
                                  + " - cannot have '(', ')', or ','")

        if func_code.co_argcount == 1:
            func_dict = self.norm_func_dict[which_type]
            func_dict[name] = user_func
            return

        first_aname = func_code.co_varnames[0]
        if first_aname in kwargs:
            raise InvalidFunction("First argument '{0!s}' of function {1!r}".format(first_aname,
                                                                                    user_func)
                                  + " ({0!s}) may not be in kwargs {1!r}".format(name,kwargs))

        evolved_param_names = func_code.co_varnames[1:func_code.co_argcount]
        func_names_set = set(evolved_param_names)
        kwargs_names_set = set(kwargs.keys())
        
        missing1 = func_names_set - kwargs_names_set
        if missing1:
            raise InvalidFunction("Function {0!r} ({1!s}) has arguments '".format(user_func,
                                                                                  name)
                                  + "{0!r}' not in kwargs {1!r}".format(missing1,
                                                                        kwargs_names_set))
        missing2 = kwargs_names_set - func_names_set
        if missing2:
            raise InvalidFunction("Function {0!r} ({1!s}) lacks arguments '".format(user_func,
                                                                                    name)
                                  + "{0!r}' in kwargs {1!r}".format(missing2,kwargs))

        func_dict = self.multiparam_func_dict[which_type]
        func_dict[name] = MultiParameterFunction(name=name, which_type=which_type,
                                                 user_func=user_func,
                                                 evolved_param_names=evolved_param_names,
                                                 evolved_param_dicts=kwargs)

