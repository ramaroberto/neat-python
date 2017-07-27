"""Deals with the attributes (variable parameters) of genes"""
#from __future__ import print_function

from copy import copy
from random import choice, gauss, random, uniform
from sys import version_info

from neat.config import ConfigParameter
from neat.six_util import iterkeys, iteritems

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

# MYPY - really needs a way to tell it that a given attribute is created dynamically...

if MYPY: # pragma: no cover
    from neat.multiparameter import MultiParameterFunctionInstance # pylint: disable=unused-import
    from neat.genome import DefaultGenomeConfig # pylint: disable=unused-import

if version_info.major > 2:
    unicode = str # pylint: disable=redefined-builtin


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.


class BaseAttribute(object):
    """Superclass for the type-specialized attribute subclasses, used by genes."""
    _config_items = {} # type: Dict[str, List[Any]]

    def __init__(self,
                 name, # type: str
                 **default_dict # type: Any
                 ):
        # type: (...) -> None
        self.name = name
        for n, default in iteritems(default_dict):
            self._config_items[n] = [self._config_items[n][0], default]
        for n in iterkeys(self._config_items):
            setattr(self, n + "_name", self.config_item_name(n))

    def config_item_name(self, config_item_base_name): # type: (str) -> str
        return "{0}_{1}".format(self.name, config_item_base_name)

    def get_config_params(self): # type: () -> List[ConfigParameter]
        return [ConfigParameter(self.config_item_name(n),
                                self._config_items[n][0],
                                self._config_items[n][1])
                for n in iterkeys(self._config_items)]

class FloatAttribute(BaseAttribute):
    """
    Class for numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    _config_items = {"init_mean": [float, None],
                     "init_stdev": [float, None],
                     "init_type": [str, 'gaussian'],
                     "replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None]} # type: Dict[str, List[Any]]

    def clamp(self, value, config): # type: (float, KnownConfig) -> float
        min_value = getattr(config, self.min_value_name) # type: ignore # type: float
        max_value = getattr(config, self.max_value_name) # type: ignore # type: float
        return max(min(value, max_value), min_value)

    def init_value(self, config): # type: (KnownConfig) -> float
        mean = getattr(config, self.init_mean_name) # type: ignore # type: float
        stdev = getattr(config, self.init_stdev_name) # type: ignore # type: float
        init_type = getattr(config, self.init_type_name).lower() # type: ignore # type: str

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name), # type: ignore
                            (mean-(2.0*stdev))) # type: float
            max_value = min(getattr(config, self.max_value_name), # type: ignore
                            (mean+(2.0*stdev))) # type: float
            return uniform(min_value, max_value)

        raise RuntimeError(
            "Unknown init_type {!r} for {!s}".format(getattr(config,
                                                             self.init_type_name), # type: ignore
                                                     self.init_type_name)) # type: ignore

    def mutate_value(self, value, config): # type: (float, KnownConfig) -> float
         # mutate_rate is usually no lower than replace_rate,
         # and frequently higher - so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name) # type: ignore # type: float

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name) # type: ignore # type: float
            return self.clamp(value + gauss(0.0, mutate_power), config)

        replace_rate = getattr(config, self.replace_rate_name) # type: ignore # type: float

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

    def validate(self, config):
        pass


class BoolAttribute(BaseAttribute):
    """Class for boolean attributes such as whether a connection is enabled or not."""
    _config_items = {"default": [str, None],
                     "mutate_rate": [float, None],
                     "rate_to_true_add": [float, 0.0],
                     "rate_to_false_add": [float, 0.0]} # type: Dict[str, List[Any]]

    def init_value(self, config): # type: (KnownConfig) -> bool
        default = str(getattr(config, self.default_name)).lower() # type: ignore # type: str

        if default in ('1', 'on', 'yes', 'true'):
            return True
        elif default in ('0', 'off', 'no', 'false'):
            return False
        elif default in ('random', 'none'):
            return bool(random() < 0.5)

        raise RuntimeError("Unknown default value {!r} for {!s}".format(default,
                                                                        self.name))

    def mutate_value(self, value, config): # type: (bool, KnownConfig) -> bool
        mutate_rate = getattr(config, self.mutate_rate_name) # type: ignore # type: float

        if value:
            mutate_rate += getattr(config, self.rate_to_false_add_name) # type: ignore
        else:
            mutate_rate += getattr(config, self.rate_to_true_add_name) # type: ignore

        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                # NOTE: we choose a random value here so that the mutation rate has the
                # same exact meaning as the rates given for the string and bool
                # attributes (the mutation operation *may* change the value but is not
                # guaranteed to do so).
                return random() < 0.5

        return value

    def validate(self, config):
        pass


class StringAttribute(BaseAttribute):
    """
    Class for string attributes such as the aggregation function of a node,
    which are selected from a list of options.
    """
    _config_items = {"default": [str, 'random'],
                     "options": [list, None],
                     "mutate_rate": [float, None]} # type: Dict[str, List[Any]]

    def init_value(self, config): # type: (KnownConfig) -> str
        default = getattr(config, self.default_name) # type: ignore # type: str

        if default.lower() in ('none','random'):
            options = getattr(config, self.options_name) # type: ignore # type: List[str]
            return choice(options)

        return default

    def mutate_value(self, value, config): # type: (str, KnownConfig) -> str
        mutate_rate = getattr(config, self.mutate_rate_name) # type: ignore # type: float

        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                options = getattr(config, self.options_name) # type: ignore # type: List[str]
                return choice(options)

        return value

    def validate(self, config):
        pass

class FuncAttribute(BaseAttribute):
    """
    Handle attributes that may be simple strings
    or may be functions needing multiparameter handling.
    """
    _config_items = copy(StringAttribute._config_items) # type: Dict[str, List[Any]]

    def init_value(self, config): # type: (DefaultGenomeConfig) -> Union[str, MultiParameterFunctionInstance]
        default = getattr(config,
                          self.default_name) # type: ignore # type: Union[str, MultiParameterFunctionInstance, None]

        if default in (None, 'random'):
            options = getattr(config,
                              self.options_name) # type: ignore # type: List[Union[str, MultiParameterFunctionInstance]]
            default = choice(options)
            if MYPY: # pragma: no cover
                default = cast(Union[str, MultiParameterFunctionInstance], default)

        if hasattr(default, 'init_value'):
            default.init_value(config)
        elif not isinstance(default, (str, unicode)):
            raise RuntimeError("Unknown what to do with default {0!r} for {1!s}".format(default,
                                                                                        self.name))
        elif hasattr(config, 'multiparameterset'):
            multiparam = config.multiparameterset

            if multiparam.is_multiparameter(default, self.name):
                default = multiparam.init_multiparameter(default, self, config)

        return default


    def mutate_value(self,
                     value, # type: Union[str, MultiParameterFunctionInstance]
                     config # type: DefaultGenomeConfig
                     ):
        # type: (...) -> Union[str, MultiParameterFunctionInstance]
        mutate_rate = getattr(config, self.mutate_rate_name) # type: ignore # type: float

        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                options = getattr(config, self.options_name) # type: ignore
                value = choice(options)

        if hasattr(value, 'mutate_value'):
            #print("Accessing mutate_value function of {!r}".format(value))
            value.mutate_value(config) # type: ignore
        elif not isinstance(value, (str, unicode)):
            raise RuntimeError("Unknown what to do with value {0!r} for {1!s}".format(value,
                                                                                      self.name))
        elif hasattr(config, 'multiparameterset'):
            multiparam = config.multiparameterset
            if multiparam.is_multiparameter(value, self.name):
                value = multiparam.init_multiparameter(value, self, config)

        return value

