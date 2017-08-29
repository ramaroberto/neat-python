"""Does general configuration parsing; used by other classes for their configuration."""
from __future__ import print_function

import os
#import sys
import warnings

try:
    from configparser import ConfigParser, Error
except ImportError:
    from ConfigParser import Error, SafeConfigParser as ConfigParser

from neat.repr_util import check_if_file_object
from neat.six_util import iterkeys

class ConfigParameter(object):
    """Contains information about one configuration item."""
    def __init__(self, name, value_type, default=None, default_ok=False):
        self.name = name
        self.value_type = value_type
        self.default = default
        self.default_ok = default_ok
        self.used_default = False

    def __repr__(self):
        if self.default is None:
            return "ConfigParameter({0!r}, {1!r})".format(self.name,
                                                          self.value_type)
        if self.default_ok:
            return "ConfigParameter({0!r}, {1!r}, {2!r}, default_ok=True)".format(self.name,
                                                                                  self.value_type,
                                                                                  self.default)
        return "ConfigParameter({0!r}, {1!r}, {2!r})".format(self.name,
                                                             self.value_type,
                                                             self.default)

    def parse(self, section, config_parser):
        if int == self.value_type:
            return config_parser.getint(section, self.name)
        if bool == self.value_type:
            return config_parser.getboolean(section, self.name)
        if float == self.value_type:
            return config_parser.getfloat(section, self.name)
        if list == self.value_type:
            v = config_parser.get(section, self.name)
            return v.split(" ")
        if str == self.value_type:
            return config_parser.get(section, self.name)

        raise RuntimeError("Unexpected configuration type: "
                           + repr(self.value_type))

    def interpret(self, config_dict):
        """
        Converts the config_parser output into the proper type,
        supplies defaults if available and needed, and checks for some errors.
        """
        value = config_dict.get(self.name)
        if value is None:
            if self.default is None:
                raise RuntimeError('Missing configuration item: ' + self.name)
            else:
                if not self.default_ok:
                    warnings.warn("Using default {0!r} for '{1}'".format(
                        self.default, self.name),
                                  DeprecationWarning)
                self.used_default = True
                if (str != self.value_type) and isinstance(self.default, self.value_type):
                    return self.default
                else:
                    value = self.default

        try:
            if str == self.value_type:
                return str(value)
            if int == self.value_type:
                return int(value)
            if bool == self.value_type:
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
                else:
                    raise RuntimeError(self.name + " must be True or False")
            if float == self.value_type:
                return float(value)
            if list == self.value_type:
                return value.split(" ")
        except Exception:
            raise RuntimeError(
                "Error interpreting config item '{0}' with value {1!r} and type {2}".format(
                self.name, value, self.value_type))

        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))

    def format(self, value):
        if list == self.value_type:
            return " ".join(value)
        return str(value)


def write_pretty_params(f, config, params):
    """
    Writes out current config parameters to file object f in a
    format suitable for to be read back in as a configuration file.
    """
    param_names = [p.name for p in params]
    longest_name_len = max(len(name) for name in param_names)
    param_names.sort()
    params = dict((p.name, p) for p in params)

    param_names_used_default = [name for name in param_names
                                if (params[name].used_default
                                    and not params[name].default_ok)]
    param_names_default_ok = [name for name in param_names
                              if (params[name].used_default
                                  and params[name].default_ok)]
    param_names_not_defaulted = [name for name in param_names
                                 if not params[name].used_default]

    if param_names_not_defaulted:
        for name in param_names_not_defaulted:
            p = params[name]
            f.write('{} = {}\n'.format(p.name.ljust(longest_name_len),
                                       p.format(getattr(config, p.name))))

    if param_names_default_ok:
        f.write('\n# Used expected default:\n')
        for name in param_names_default_ok:
            p = params[name]
            f.write('{} = {}\n'.format(p.name.ljust(longest_name_len),
                                       p.format(getattr(config, p.name))))

    if param_names_used_default:
        f.write('\n# Used possibly-unexpected default:\n')
        for name in param_names_used_default:
            p = params[name]
            f.write('{} = {}\n'.format(p.name.ljust(longest_name_len),
                                       p.format(getattr(config, p.name))))


class UnknownConfigItemError(NameError):
    """Error for unknown configuration option - partially to catch typos."""
    pass

class DefaultClassConfig(object):
    """
    Replaces at least some boilerplate configuration code
    for reproduction, species_set, and stagnation classes.
    """

    def __init__(self, param_dict, param_list):
        self._params = param_list
        param_list_names = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict))
            param_list_names.append(p.name)
        unknown_list = [x for x in iterkeys(param_dict) if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError("Unknown configuration item {!s}".format(unknown_list[0]))

    @classmethod
    def write_config(cls, f, config):
        # pylint: disable=protected-access
        write_pretty_params(f, config, config._params)


class Config(object):
    """A simple container for user-configurable parameters of NEAT."""

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        parameters = ConfigParser()

        if check_if_file_object(filename):
            if hasattr(parameters, 'read_file'):
                parameters.read_file(filename)
            else:
                parameters.readfp(filename)
        else:
            if not os.path.isfile(filename):
                raise Exception('No such config file: ' + os.path.abspath(filename))

            with open(filename) as f:
                if hasattr(parameters, 'read_file'):
                    parameters.read_file(f)
                else:
                    parameters.readfp(f)

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                    if getattr(self, p.name) is None:
                        setattr(self, p.name, p.default)
                        p.used_default=True
                        if not p.default_ok:
                            warnings.warn("Using default {!r} for '{!s}'".format(
                                p.default, p.name),
                                          DeprecationWarning)
                except (Error, RuntimeError):
                    setattr(self, p.name, p.default)
                    p.used_default = True
                    if not p.default_ok:
                        warnings.warn("Using default {!r} for '{!s}'".format(p.default, p.name),
                                      DeprecationWarning)
                if getattr(self, p.name) != p.default:
                    p.used_default = False # Why needed???
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in iterkeys(param_dict) if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError(
                "Unknown (section 'NEAT') configuration item {!s}".format(unknown_list[0]))


        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))
        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

    def save(self, filename):
        if check_if_file_object(filename):
            self._save(filename)
        else:
            with open(filename, 'w') as f:
                self._save(f)

    def _save(self, f):
        f.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
        f.write('# or the experiment itself.  This is the only required section.\n')
        f.write('[NEAT]\n')
        write_pretty_params(f, self, self.__params)
        
        f.write('\n[{0}]\n'.format(self.genome_type.__name__))
        self.genome_type.write_config(f, self.genome_config)
        
        f.write('\n[{0}]\n'.format(self.species_set_type.__name__))
        self.species_set_type.write_config(f, self.species_set_config)
        
        f.write('\n[{0}]\n'.format(self.stagnation_type.__name__))
        self.stagnation_type.write_config(f, self.stagnation_config)
        
        f.write('\n[{0}]\n'.format(self.reproduction_type.__name__))
        self.reproduction_type.write_config(f, self.reproduction_config)
