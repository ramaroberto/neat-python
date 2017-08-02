from __future__ import print_function
import os
import sys
from unittest.case import SkipTest

if ((sys.version_info.major < 3) or
    ((sys.version_info.major == 3) and (sys.version_info.minor < 3))):
    raise SkipTest("Need Python 3.3+ for mypy to run")

try:
    from mypy import api
except ImportError:
    raise SkipTest("Unable to access mypy (pypy3?)")

def get_file_list(neat_path):
    # all but distributed.py
    init_file_list = """./__init__.py
    ./activations.py
    ./aggregations.py
    ./attributes.py
    ./checkpoint.py
    ./config.py
    ./ctrnn
    ./genes.py
    ./genome.py
    ./graphs.py
    ./iznn
    ./math_util.py
    ./multiparameter.py
    ./mypy_util.py
    ./nn
    ./parallel.py
    ./population.py
    ./reporting.py
    ./reproduction.py
    ./six_util.py
    ./species.py
    ./stagnation.py
    ./statistics.py
    ./threaded.py""".split()
    file_iterator = map(os.path.normpath,init_file_list)
    use_file_list = [os.path.normpath(os.path.join(neat_path,x)) for x in file_iterator]
    return use_file_list

def do_mypy(config_file):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)

    neat_path = os.path.normpath(os.path.join(local_dir, '../neat'))

    mypy_input_list = ["--config-file", config_path] + get_file_list(neat_path)

    mypy_results = api.run(mypy_input_list)

    if mypy_results[2] != 0:
        if mypy_results[0]:
            print("\nMypy stdout report:", file=sys.stderr)
            print(mypy_results[0], file=sys.stderr, flush=True)
        if mypy_results[1]:
            print("\nMypy stderr report:", file=sys.stderr)
            print(mypy_results[1], file=sys.stderr, flush=True)
        if not (mypy_results[0] or mypy_results[1]):
            print("\nMypy - no report(s) but status is {:n}".format(mypy_results[2]),
                  file=sys.stderr)
            print("Mypy input list was {!r}".format(mypy_input_list),
                  file=sys.stderr, flush=True)
        raise Exception("Mypy error (status {0:n}) with config file {1!s}".format(mypy_results[2],
                                                                                  config_path))
    elif mypy_results[1] and not mypy_results[1].isspace():
        print("\nMypy - status 0 but error report:", file=sys.stderr)
        print(mypy_results[1], file=sys.stderr, flush=True)
        raise Exception("Mypy error with config file {!s}".format(config_path))

def test_mypy_wrapper():
    """Do testing vs mypy (3.3+ and possibly 2.7)"""
    do_mypy('testing_mypy.ini')

    minor_for_extra = 5
    if 'TRAVIS' in os.environ:
        minor_for_extra = 6

    if ((sys.version_info.major == 3) and (sys.version_info.minor >= minor_for_extra) and (sys.version_info.releaselevel == 'final')):
        print("Also doing mypy testing vs 2.7", file=sys.stderr)
        do_mypy('testing_mypy_2.7.ini')

if __name__ == '__main__':
    test_mypy_wrapper()
