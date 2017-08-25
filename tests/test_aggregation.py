import os
import warnings

import neat
from neat import aggregations, multiparameter

warnings.simplefilter('default')

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

class NotAlmostEqualException(AssertionError):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException("{0!r} !~= {1!r}".format(float(a), float(b)))

def test_sum():
    assert aggregations.sum_aggregation([1.0,2.0,0.5]) == 3.5
    assert aggregations.sum_aggregation([1.0,0.5,0.0]) == 1.5
    assert aggregations.sum_aggregation([1.0,-1.0,0.0]) == 0.0

def test_product():
    assert aggregations.product_aggregation([1.0,2.0,0.5]) == 1.0
    assert aggregations.product_aggregation([1.0,0.5,0.0]) == 0.0
    assert aggregations.product_aggregation([4.0,2.0,-1.0]) == -8.0
    
def test_max():
    assert aggregations.max_aggregation([0.0,1.0,2.0]) == 2.0
    assert aggregations.max_aggregation([0.0,-1.0,-2.0]) == 0.0

def test_min():
    assert aggregations.min_aggregation([0.0,1.0,2.0]) == 0.0
    assert aggregations.min_aggregation([0.0,-1.0,-2.0]) == -2.0

def test_maxabs():
    assert aggregations.maxabs_aggregation([0.0,1.0,2.0]) == 2.0
    assert aggregations.maxabs_aggregation([0.0,-1.0,-2.0]) == -2.0

def test_median():
    assert aggregations.median_aggregation([-1.0, -1.0, -1.0, -0.5, -0.5]) == -1.0
    assert aggregations.median_aggregation([-1.0, -1.0, -0.5, -0.5, 0.0]) == -0.5
    assert aggregations.median_aggregation([-1.0, -1.0, 0.0, 0.5, 0.5]) == 0.0
    assert aggregations.median_aggregation([-1.0, -1.0, 0.5, 0.5, 1.0]) == 0.5
    assert aggregations.median_aggregation([-0.5, -0.5, 1.0, 1.0, 1.0]) == 1.0
    assert aggregations.median_aggregation([-2.0, -1.0, -0.5, -0.5]) == -0.75
    assert aggregations.median_aggregation([-2.0, -1.0, -0.5, 0.5]) == -0.75
    assert aggregations.median_aggregation([-2.0, -1.0, 0.5, 0.5]) == -0.25
    assert aggregations.median_aggregation([-2.0, -0.5, 0.0, 0.5]) == -0.25
    assert aggregations.median_aggregation([-2.0, -0.5, -0.5, 0.0]) == -0.5
    assert aggregations.median_aggregation([-2.0, -0.5, -0.5, 1.0]) == -0.5
    assert aggregations.median_aggregation([-2.0, -0.5, 0.5, 1.0]) == 0.0
    assert aggregations.median_aggregation([-1.0, -0.5, 0.5, 1.0]) == 0.0
    assert aggregations.median_aggregation([-2.0, 0.0, 0.5, 0.5]) == 0.25
    assert aggregations.median_aggregation([-1.0, 0.0, 0.5, 0.5]) == 0.25
    assert aggregations.median_aggregation([-2.0, 0.5, 0.5, 1.0]) == 0.5
    assert aggregations.median_aggregation([-1.0, 0.5, 0.5, 1.0]) == 0.5
    assert aggregations.median_aggregation([-0.5, 0.5, 1.0, 1.0]) == 0.75
    assert aggregations.median_aggregation([-0.5, 0.5, 1.0, 2.0]) == 0.75

def test_mean():
    assert aggregations.mean_aggregation([0.0,1.0,2.0]) == 1.0
    assert aggregations.mean_aggregation([0.0,-1.0,-2.0]) == -1.0

def test_tmean():
    assert_almost_equal(aggregations.tmean_aggregation([-1.0, -1.0, -1.0, -0.5, -0.5]),
                        -1*aggregations.tmean_aggregation([-0.5, 0.5, 1.0, 1.0, 1.0]))
    assert aggregations.tmean_aggregation([-1.0, -1.0, -1.0, 0.5, 0.5]) == -0.75
    assert_almost_equal(aggregations.tmean_aggregation([-1.0, -1.0, -0.5, -0.5, 0.0]),
                        -1*aggregations.tmean_aggregation([-1.0, 0.5, 0.5, 1.0, 1.0]))
    assert aggregations.tmean_aggregation([-1.0, -1.0, -0.5, 0.0, 0.5]) == -0.5
    assert_almost_equal(aggregations.tmean_aggregation([-1.0, -1.0, -0.5, 0.5, 1.0]),
                        -1*aggregations.tmean_aggregation([-1.0, -0.5, 0.5, 1.0, 1.0]))
    assert_almost_equal(aggregations.tmean_aggregation([-1.0, -1.0, 0.0, 0.5, 0.5]),
                        -1*aggregations.tmean_aggregation([-1.0, 0.0, 0.0, 0.5, 0.5]))
    assert aggregations.tmean_aggregation([-2.0, -1.0, -0.5, -0.5]) == -0.75
    assert aggregations.tmean_aggregation([-2.0, -1.0, -0.5, 0.5]) == -0.75
    assert aggregations.tmean_aggregation([-2.0, -0.5, -0.5, 0.0]) == -0.5
    assert aggregations.tmean_aggregation([-2.0, -0.5, -0.5, 1.0]) == -0.5
    assert aggregations.tmean_aggregation([-2.0, -1.0, 0.5, 0.5]) == -0.25
    assert aggregations.tmean_aggregation([-2.0, -0.5, 0.0, 0.5]) == -0.25
    assert aggregations.tmean_aggregation([-2.0, -0.5, 0.5, 1.0]) == 0.0
    assert aggregations.tmean_aggregation([-1.0, -0.5, 0.5, 1.0]) == 0.0
    assert aggregations.tmean_aggregation([-2.0, 0.0, 0.5, 0.5]) == 0.25
    assert aggregations.tmean_aggregation([-1.0, 0.0, 0.5, 0.5]) == 0.25
    assert aggregations.tmean_aggregation([-2.0, 0.5, 0.5, 1.0]) == 0.5
    assert aggregations.tmean_aggregation([-1.0, 0.5, 0.5, 1.0]) == 0.5
    assert aggregations.tmean_aggregation([-0.5, 0.5, 1.0, 1.0]) == 0.75
    assert aggregations.tmean_aggregation([-0.5, 0.5, 1.0, 2.0]) == 0.75

def test_multiparam_tmean():
    assert aggregations.multiparam_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.25) == -0.75
    assert aggregations.multiparam_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.0) == -1.0
    assert aggregations.multiparam_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.5) == -0.75
    assert aggregations.multiparam_tmean_aggregation([-2.0, -1.0, 0.5, 0.5], 0.25) == -0.25
    assert aggregations.multiparam_tmean_aggregation([-2.0, -1.0, 0.5, 0.5], 0.0) == -0.5
    assert aggregations.multiparam_tmean_aggregation([-2.0, -1.0, 0.5, 0.5], 0.5) == -0.25
    assert aggregations.multiparam_tmean_aggregation([-2.0, -0.5, 0.5, 1.0], 0.25) == 0.0
    assert aggregations.multiparam_tmean_aggregation([-2.0, -0.5, 0.5, 1.0], 0.5) == 0.0
    assert aggregations.multiparam_tmean_aggregation([-2.0, 0.0, 0.5, 0.5], 0.25) == 0.25
    assert aggregations.multiparam_tmean_aggregation([-2.0, 0.0, 0.5, 0.5], 0.5) == 0.25
    assert aggregations.multiparam_tmean_aggregation([-2.0, 0.5, 0.5, 1.0], 0.25) == 0.5
    assert aggregations.multiparam_tmean_aggregation([-2.0, 0.5, 0.5, 1.0], 0.5) == 0.5
    assert aggregations.multiparam_tmean_aggregation([-0.5, 0.5, 1.0, 1.0], 0.25) == 0.75
    assert aggregations.multiparam_tmean_aggregation([-0.5, 0.5, 1.0, 1.0], 0.5) == 0.75
    assert aggregations.multiparam_tmean_aggregation([0.5, 0.5, 1.0, 2.0], 0.0) == 1.0
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-1.0, -0.5, -0.5, 1.0], 0.125),
                        -1*aggregations.multiparam_tmean_aggregation([-1.0, 0.5, 0.5, 1.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-1.0, -1.0, -0.5, 0.5], 0.125),
                        -1*aggregations.multiparam_tmean_aggregation([-0.5, 0.5, 1.0, 1.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-2.0, -0.5, -0.5, 0.0], 0.125),
                        -1*aggregations.multiparam_tmean_aggregation([0.0, 0.5, 0.5, 2.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.125),
                        -1*aggregations.multiparam_tmean_aggregation([0.5, 0.5, 1.0, 2.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-2.0, -1.0, 0.5, 0.5], 0.125),
                        aggregations.multiparam_tmean_aggregation([-2.0, -0.5, 0.0, 0.5], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-2.0, -0.5, 0.5, 1.0], 0.125),
                        aggregations.multiparam_tmean_aggregation([-1.0, -0.5, -0.5, 2.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-2.0, 0.0, 0.5, 0.5], 0.125),
                        aggregations.multiparam_tmean_aggregation([-0.5, -0.5, 0.0, 1.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-1.0, 0.0, 0.5, 0.5], 0.125),
                        aggregations.multiparam_tmean_aggregation([-0.5, -0.5, 0.0, 2.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-2.0, 0.5, 0.5, 1.0], 0.125),
                        aggregations.multiparam_tmean_aggregation([-1.0, -0.5, 0.5, 2.0], 0.125))
    assert_almost_equal(aggregations.multiparam_tmean_aggregation([-0.5, -0.5, 1.0, 2.0], 0.125),
                        aggregations.multiparam_tmean_aggregation([-0.5, 0.0, 0.5, 2.0], 0.125))


def test_max_median_min():
    assert aggregations.max_median_min_aggregation([0.0,1.0,2.0],1.0) == 2.0
    assert aggregations.max_median_min_aggregation([0.0,-1.0,-2.0],1.0) == 0.0
    assert aggregations.max_median_min_aggregation([0.0,1.0,2.0],0.0) == 1.0
    assert aggregations.max_median_min_aggregation([-10.0,1.0,3.0,10.0],0.0) == 2.0
    assert aggregations.max_median_min_aggregation([0.0,1.0,2.0],-1.0) == 0.0
    assert aggregations.max_median_min_aggregation([0.0,-1.0,-2.0],-1.0) == -2.0

def test_maxabs_mean():
    assert aggregations.maxabs_mean_aggregation([0.0,1.0,2.0],1.0) == 2.0
    assert aggregations.maxabs_mean_aggregation([0.0,-1.0,-2.0],1.0) == -2.0
    assert aggregations.maxabs_mean_aggregation([0.0,1.0,2.0],0.0) == 1.0
    assert aggregations.maxabs_mean_aggregation([0.0,-1.0,-2.0],0.0) == -1.0

def test_maxabs_tmean():
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.0) == -1.0
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], -1.0) == -0.75
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], 1.0) == -2.0
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.5) == -1.5
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, -0.5, 0.5], 0.5) == -1.375
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, 0.5, 0.5], 0.0) == -0.5
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, 0.5, 0.5], -1.0) == -0.25
    assert aggregations.maxabs_tmean_aggregation([-2.0, -1.0, 0.5, 0.5], 0.5) == -1.25
    assert aggregations.maxabs_tmean_aggregation([-2.0, -0.5, -0.5, 0.0], -1.0) == -0.5
    assert aggregations.maxabs_tmean_aggregation([-2.0, -0.5, 0.5, 1.0], -1.0) == 0.0
    assert aggregations.maxabs_tmean_aggregation([-2.0, -0.5, 0.5, 1.0], 0.5) == -1.125
    assert aggregations.maxabs_tmean_aggregation([-2.0, 0.0, 0.5, 0.5], -1.0) == 0.25
    assert aggregations.maxabs_tmean_aggregation([-2.0, 0.5, 0.5, 1.0], -1.0) == 0.5
    assert aggregations.maxabs_tmean_aggregation([-2.0, 0.5, 0.5, 1.0], 0.5) == -1.0
    assert aggregations.maxabs_tmean_aggregation([-1.0, -1.0, -0.5, -0.5], 0.5) == -0.875
    assert aggregations.maxabs_tmean_aggregation([-1.0, -1.0, 0.5, 0.5], 0.5) == -0.625
    assert aggregations.maxabs_tmean_aggregation([-1.0, -0.5, -0.5, 2.0], 1.0) == 2.0
    assert aggregations.maxabs_tmean_aggregation([-1.0, -0.5, -0.5, 2.0], 0.5) == 1.0
    assert aggregations.maxabs_tmean_aggregation([-1.0, -0.5, 0.5, 2.0], 0.5) == 1.125
    assert aggregations.maxabs_tmean_aggregation([-1.0, 0.5, 0.5, 1.0], 0.5) == -0.375
    assert aggregations.maxabs_tmean_aggregation([-1.0, 0.5, 0.5, 2.0], 0.5) == 1.25
    assert aggregations.maxabs_tmean_aggregation([-0.5, -0.5, 1.0, 1.0], 0.5) == 0.625
    assert aggregations.maxabs_tmean_aggregation([-0.5, 0.5, 1.0, 1.0], -1.0) == 0.75
    assert aggregations.maxabs_tmean_aggregation([-0.5, 0.5, 1.0, 2.0], 0.5) == 1.375
    assert aggregations.maxabs_tmean_aggregation([0.0, 0.0, 0.5, 0.5], 0.5) == 0.375
    assert aggregations.maxabs_tmean_aggregation([0.5, 0.5, 1.0, 1.0], 0.5) == 0.875
    assert aggregations.maxabs_tmean_aggregation([0.5, 0.5, 1.0, 2.0], 0.5) == 1.5


def test_sum_mean():
    assert aggregations.sum_mean_aggregation([1.0,2.0,0.5], 1.0) == 3.5
    assert aggregations.sum_mean_aggregation([1.0,-1.0,0.0], 1.0) == 0.0
    assert aggregations.sum_mean_aggregation([0.0,1.0,2.0], 0.0) == 1.0
    assert aggregations.sum_mean_aggregation([0.0,-1.0,-2.0], 0.0) == -1.0
    assert aggregations.sum_mean_aggregation([-2.0, -1.0, -0.5, -0.5], 1.0) == -4.0
    assert aggregations.sum_mean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.75) == -3.25
    assert aggregations.sum_mean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.5) == -2.5
    assert aggregations.sum_mean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.25) == -1.75
    assert aggregations.sum_mean_aggregation([-2.0, -1.0, -0.5, -0.5], 0.0) == -1.0

def test_product_mean():
    assert aggregations.product_mean_aggregation([1.0,2.0,0.5], 1.0, True) == 1.0
    assert aggregations.product_mean_aggregation([1.0,0.5,0.0], 1.0, True) == 0.0
    assert aggregations.product_mean_aggregation([2.0,2.0], 0.0, False) == 2.0
    assert aggregations.product_mean_aggregation([4.0,2.0,1.0], 0.0, False) == 2.0
    assert aggregations.product_mean_aggregation([-2.0, -1.0, -0.5, -0.5], 1.0, False) == 0.5
    assert aggregations.product_mean_aggregation([-2.0, -1.0, -0.5, -0.5], 1.0, True) == -0.5
    assert aggregations.product_mean_aggregation([-1.0, -1.0, -0.5, -0.5], 1.0, False) == 0.25
    assert aggregations.product_mean_aggregation([-1.0, -1.0, -0.5, -0.5], 1.0, True) == -0.25
    assert aggregations.product_mean_aggregation([-2.0, -0.5, 0.0, 0.5], 0.5, False) == 0.0


def test_sum_product_mean():
    assert aggregations.sum_product_mean_aggregation([1.0,2.0,0.5], 1.0, 1.0, True) == 3.5
    assert aggregations.sum_product_mean_aggregation([1.0,-1.0,0.0], 1.0, 1.0, False) == 0.0
    assert aggregations.sum_product_mean_aggregation([0.0,1.0,2.0], 0.0, 1.0, True) == 1.0
    assert aggregations.sum_product_mean_aggregation([0.0,-1.0,-2.0], 0.0, 1.0, True) == -1.0
    assert aggregations.sum_product_mean_aggregation([1.0,2.0,0.5], 1.0, 0.0, False) == 1.0
    assert aggregations.sum_product_mean_aggregation([1.0,0.5,0.0], 1.0, 0.0, False) == 0.0
    assert aggregations.sum_product_mean_aggregation([2.0,2.0], 0.0, 0.0, True) == 2.0
    assert aggregations.sum_product_mean_aggregation([4.0,2.0,1.0], 0.0, 0.0, False) == 2.0

def test_sum_product():
    assert aggregations.sum_product_aggregation([1.0,2.0,0.5], 1.0) == 3.5
    assert aggregations.sum_product_aggregation([1.0,0.5,0.0], 1.0) == 1.5
    assert aggregations.sum_product_aggregation([1.0,-1.0,0.0], 1.0) == 0.0
    assert aggregations.sum_product_aggregation([1.0,2.0,0.5], 0.0) == 1.0
    assert aggregations.sum_product_aggregation([1.0,0.5,0.0], 0.0) == 0.0
    assert aggregations.sum_product_aggregation([4.0,2.0,-1.0], 0.0) == -8.0
    assert aggregations.sum_product_aggregation([-2.0, -1.0, -0.5, -0.5], 1.0) == -4.0
    assert aggregations.sum_product_aggregation([-2.0, -1.0, -0.5, -0.5], 0.75) == -2.875
    assert aggregations.sum_product_aggregation([-2.0, -1.0, -0.5, -0.5], 0.5) == -1.75
    assert aggregations.sum_product_aggregation([-2.0, -1.0, -0.5, -0.5], 0.25) == -0.625
    assert aggregations.sum_product_aggregation([-2.0, -1.0, -0.5, -0.5], 0.0) == 0.5
    assert aggregations.sum_product_aggregation([-1.0, -1.0, -0.5, -0.5], 0.75) == -2.1875
    assert aggregations.sum_product_aggregation([-1.0, -1.0, -0.5, -0.5], 0.25) == -0.5625
    assert aggregations.sum_product_aggregation([-1.0, -1.0, -0.5, -0.5], 0.0) == 0.25


def minabs_aggregation(x):
    """Not particularly useful - just a check that can load in via genome_config."""
    return min(x, key=abs)

def test_add_minabs():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.genome_config.add_aggregation('minabs', minabs_aggregation)
    assert config.genome_config.aggregation_function_defs.get('minabs') is not None
    assert config.genome_config.aggregation_function_defs['minabs'] is not None
    assert config.genome_config.aggregation_function_defs.is_valid('minabs')

def dud_function():
    return 0.0

def test_function_set():
    m = multiparameter.MultiParameterSet('aggregation')
    s = aggregations.AggregationFunctionSet(m)
    assert s.get('sum') is not None
    assert s.get('product') is not None
    assert s.get('max') is not None
    assert s.get('min') is not None
    assert s.get('maxabs') is not None
    assert s.get('median') is not None
    assert s.get('mean') is not None
    assert s.get('tmean') is not None
    assert m.get_MPF('multiparam_tmean', 'aggregation') is not None
    assert m.get_MPF('max_median_min', 'aggregation') is not None
    assert m.get_MPF('maxabs_mean', 'aggregation') is not None
    assert m.get_MPF('maxabs_tmean', 'aggregation') is not None
    assert m.get_MPF('sum_mean', 'aggregation') is not None
    assert m.get_MPF('product_mean', 'aggregation') is not None
    assert m.get_MPF('sum_product_mean', 'aggregation') is not None
    assert m.get_MPF('sum_product', 'aggregation') is not None

    assert s.is_valid('sum')
    assert s.is_valid('product')
    assert s.is_valid('max')
    assert s.is_valid('min')
    assert s.is_valid('maxabs')
    assert s.is_valid('median')
    assert s.is_valid('mean')
    assert s.is_valid('tmean')
    assert s.is_valid('multiparam_tmean')
    assert s.is_valid('max_median_min')
    assert s.is_valid('maxabs_mean')
    assert s.is_valid('maxabs_tmean')
    assert s.is_valid('sum_mean')
    assert s.is_valid('product_mean')
    assert s.is_valid('sum_product_mean')
    assert s.is_valid('sum_product')

    assert not s.is_valid('foo')

    try:
        ignored = s['foo']
    except LookupError:
        pass
    else:
        raise Exception("Should have gotten a LookupError/derived for dict lookup of 'foo'")

def test_bad_add1():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    try:
        config.genome_config.add_aggregation('1.0',1.0)
    except TypeError:
        pass
    else:
        raise Exception("Should have had a TypeError/derived for 'function' 1.0")

def test_bad_add2():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    try:
        config.genome_config.add_aggregation('dud_function',dud_function)
    except TypeError:
        pass
    else:
        raise Exception("Should have had a TypeError/derived for dud_function")

def test_get_MPF():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    assert config.genome_config.get_aggregation_MPF('max_median_min') is not None
    assert config.genome_config.get_aggregation_MPF('maxabs_mean') is not None
    assert config.genome_config.get_aggregation_MPF('sum_mean') is not None
    assert config.genome_config.get_aggregation_MPF('product_mean') is not None
    assert config.genome_config.get_aggregation_MPF('sum_product_mean') is not None
    assert config.genome_config.get_aggregation_MPF('sum_product') is not None

    try:
        ignored = config.genome_config.get_aggregation_MPF('foo')
    except LookupError:
        pass
    else:
        raise Exception("Should have had a LookupError/derived for get_aggregation_MPF 'foo'")

    sum_product_mean_MPF = config.genome_config.get_aggregation_MPF('sum_product_mean')
    new_sum_product_mean_MPF = sum_product_mean_MPF.copy_and_change(
        del_not_changed=True, new_param_dicts={'use_median':{'rate_to_true_add':0.05}})
    new2_sum_product_mean_MPF = sum_product_mean_MPF.copy_and_change(
        del_param_dicts={'use_median':None},
        new_param_dicts={'use_median':{'rate_to_true_add':0.05}})
    assert new_sum_product_mean_MPF is not None
    assert new2_sum_product_mean_MPF is not None
    assert id(sum_product_mean_MPF) != id(new_sum_product_mean_MPF)
    assert id(sum_product_mean_MPF) != id(new2_sum_product_mean_MPF)

def test_get_Evolved_MPF_simple():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    assert config.genome_config.get_aggregation_Evolved_MPF('max_median_min') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('maxabs_mean') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('sum_mean') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('product_mean') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('sum_product_mean') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('sum_product') is not None

    try:
        ignored = config.genome_config.get_aggregation_Evolved_MPF('foo')
    except LookupError:
        pass
    else:
        raise Exception(
            "Should have had a LookupError/derived for get_aggregation_Evolved_MPF('foo')")

def test_get_Evolved_MPF_complex():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    assert config.genome_config.get_aggregation_Evolved_MPF('max_median_min(0.5)') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('maxabs_mean(0.5)') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('sum_mean(0.5)') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('product_mean(0.5,True)') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('sum_product_mean(0.5,0.5,True)') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('sum_product(0.5)') is not None
    assert config.genome_config.get_aggregation_Evolved_MPF('sum_product(1)') is not None

    try:
        ignored = config.genome_config.get_aggregation_Evolved_MPF('foo(0.5)')
    except LookupError:
        pass
    else:
        raise Exception(
            "Should have had a LookupError/derived for get_aggregation_Evolved_MPF('foo(0.5)')")

    try:
        ignored = config.genome_config.get_aggregation_Evolved_MPF('maxabs_mean(0.5,0.5,0.5)')
    except RuntimeError:
        pass
    else:
        raise Exception(
            "Should have had a RuntimeError/derived for get_aggregation_Evolved_MPF('maxabs_mean(0.5,0.5,0.5)')")

    test_result = config.genome_config.get_aggregation_Evolved_MPF('product_mean(0.5,True)')
    assert str(test_result) == str(config.genome_config.get_aggregation_Evolved_MPF(str(test_result)))
    assert config.genome_config.multiparameterset.get_func(str(test_result), 'aggregation') is not None

if __name__ == '__main__':
    test_sum()
    test_product()
    test_max()
    test_min()
    test_maxabs()
    test_median()
    test_mean()
    test_tmean()
    test_multiparam_tmean()
    test_max_median_min()
    test_maxabs_mean()
    test_maxabs_tmean()
    test_sum_mean()
    test_product_mean()
    test_sum_product_mean()
    test_sum_product()
    test_add_minabs()
    test_function_set()
    test_get_MPF()
    test_get_Evolved_MPF_simple()
    test_get_Evolved_MPF_complex()
    test_bad_add1()
    test_bad_add2()
