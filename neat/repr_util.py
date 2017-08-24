"""Utility functions to help with repr formatting and extraction of useful information."""
from __future__ import print_function

import re
import warnings

from pprint import saferepr

ERR_IF_NO_MATCH = -1
WARN_IF_NO_MATCH = 0
OK_IF_NO_MATCH = 1

def _handle_no_match(partial_return, no_match, message):
    if no_match == OK_IF_NO_MATCH:
        return partial_return

    if no_match == ERR_IF_NO_MATCH:
        raise ValueError(message)

    if no_match == WARN_IF_NO_MATCH:
        warnings.warn(message)
        return partial_return

    raise ValueError("Unknown no_match {0!r} (message {1!r})".format(
        no_match, message))

name_re = re.compile(r'(\w+)$')
full_name_re = re.compile(r'(\w+)\((.+)\)$')
module_re = re.compile(r'(\w[.a-zA-Z_]*)$')
function_re = re.compile(r'\s*<?function\s+\b(\w+)\b')
partial_re = re.compile(r'\s*<?functools\.partial\s+object\b')

def extract_function_name(repr_result, start_only=True, no_match=WARN_IF_NO_MATCH):
    if start_only:
        result = function_re.match(repr_result)
    else:
        result = function_re.search(repr_result)

    if result:
        return result.group(1)

    message = "Repr_result {0!r} did not match re {1!r}".format(
        repr_result, function_re)
    return _handle_no_match(None, no_match, message)


def partial_extract_function_args(partial_function, poss_name, no_match=WARN_IF_NO_MATCH):
    partial_args = partial_function.args
    partial_kwargs = partial_function.keywords

    if partial_args:
        if partial_kwargs:
            return "functools.partial({0!s}, *{1!r}, **{2!r})".format(poss_name,
                                                                      partial_args,
                                                                      partial_kwargs)
        return "functools.partial({0!s}, *{1!r})".format(poss_name,partial_args)

    if partial_kwargs:
        return "functools.partial({0!s}, **{1!r})".format(poss_name,partial_kwargs)

    return _handle_no_match(poss_name, no_match,
                            "Partial function {0!r} (func {1!r}) had no arguments?".format(
                                partial_function, poss_name))
        

def repr_extract_function_name(function, no_match=WARN_IF_NO_MATCH, with_module=True, as_partial=True, OK_with_args=False):
    poss_name = None
    if with_module and hasattr(function, '__qualname__'):
        result = module_re.match(str(function.__qualname__))
        if result:
            return result.group(1)
    if hasattr(function, '__name__'):
        result = name_re.match(str(function.__name__))
        if result:
            poss_name = result.group(1)
        elif not as_partial:
            result = full_name_re.match(str(function.__name__))
            if result:
                if OK_with_args:
                    poss_name = str(function.__name__)
                else:
                    poss_name = result.group(1)
    if poss_name is None: # pragma: no cover
        repr_result = saferepr(function)
        result_partial = partial_re.match(repr_result)
        if result_partial:
            poss_name = repr_extract_function_name(function.func,
                                                   no_match=no_match,
                                                   with_module=with_module)
            if (poss_name is not None) and as_partial:
                return partial_extract_function_args(function, poss_name=poss_name, no_match=no_match)
            return poss_name
        else:
            poss_name = extract_function_name(repr_result, start_only=True, no_match=no_match)

    if (not with_module) or (poss_name is None):
        return poss_name

    module_name = None
    if hasattr(function, '__module__'):
        result2 = module_re.match(str(function.__module__))
        if result2:
            module_name = result2.group(1)
    if module_name is not None:
        return module_name + '.' + poss_name

    message = "Unable to get module for function {0!s} ({1!r});".format(poss_name,
        function) + " dir result is:\n\t{0!s}".format("\n\t".join(dir(function)))

    return _handle_no_match(poss_name, no_match, message)

# TODO: More extractions of names for other things, if useful
