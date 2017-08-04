from __future__ import print_function
import os

from math import modf

import neat
from neat import activations, multiparameter

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

class NotAlmostEqualException(Exception):
    pass


def assert_almost_equal(a, b):
    if abs(a - b) > 1e-6:
        max_abs = max(abs(a), abs(b))
        abs_rel_err = abs(a - b) / max_abs
        if abs_rel_err > 1e-6:
            raise NotAlmostEqualException("{0:.6f} !~= {1:.6f}".format(a, b))


def test_sigmoid():
    assert activations.sigmoid_activation(0.0) == 0.5


def test_tanh():
    assert activations.tanh_activation(0.0) == 0.0


def test_sin():
    assert activations.sin_activation(0.0) == 0.0


def test_gauss():
    assert_almost_equal(activations.gauss_activation(0.0), 1.0)
    assert_almost_equal(activations.gauss_activation(-1.0),
                        activations.gauss_activation(1.0))


def test_relu():
    assert activations.relu_activation(-1.0) == 0.0
    assert activations.relu_activation(0.0) == 0.0
    assert activations.relu_activation(1.0) == 1.0


def test_softplus():
    assert_almost_equal(activations.softplus_activation(-5.0),0.0)
    assert 0.0 < activations.softplus_activation(0.0) < 0.25
    assert_almost_equal(activations.softplus_activation(5.0),5.0)


def test_identity():
    assert activations.identity_activation(-1.0) == -1.0
    assert activations.identity_activation(0.0) == 0.0
    assert activations.identity_activation(1.0) == 1.0


def test_clamped():
    assert activations.clamped_activation(-2.0) == -1.0
    assert activations.clamped_activation(-1.0) == -1.0
    assert activations.clamped_activation(0.0) == 0.0
    assert activations.clamped_activation(1.0) == 1.0
    assert activations.clamped_activation(2.0) == 1.0


def test_inv():
    assert activations.inv_activation(1.0) == 1.0
    assert activations.inv_activation(0.5) == 2.0
    assert activations.inv_activation(2.0) == 0.5
    assert activations.inv_activation(0.0) == 0.0


def test_log():
    assert activations.log_activation(1.0) == 0.0


def test_exp():
    assert activations.exp_activation(0.0) == 1.0


def test_abs():
    assert activations.abs_activation(-1.0) == 1.0
    assert activations.abs_activation(0.0) == 0.0
    assert activations.abs_activation(-1.0) == 1.0


def test_hat():
    assert activations.hat_activation(-1.0) == 0.0
    assert activations.hat_activation(0.0) == 1.0
    assert activations.hat_activation(1.0) == 0.0


def test_square():
    assert activations.square_activation(-1.0) == 1.0
    assert activations.square_activation(-0.5) == 0.25
    assert activations.square_activation(0.0) == 0.0
    assert activations.square_activation(0.5) == 0.25
    assert activations.square_activation(1.0) == 1.0


def test_cube():
    assert activations.cube_activation(-1.0) == -1.0
    assert activations.cube_activation(-0.5) == -0.125
    assert activations.cube_activation(0.0) == 0.0
    assert activations.cube_activation(0.5) == 0.125
    assert activations.cube_activation(1.0) == 1.0

def plus_activation(x):
    """ Not useful - just a check. """
    return abs(x+1)

def test_add_plus():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.genome_config.add_activation('plus', plus_activation)
    assert config.genome_config.activation_defs.get('plus') is not None
    assert config.genome_config.activation_defs.is_valid('plus')

def dud_function():
    return 0.0

def test_multiparam_relu():
    assert activations.multiparam_relu_activation(1.0,1.0) == 1.0
    assert activations.multiparam_relu_activation(0.0,1.0) == 0.0
    assert activations.multiparam_relu_activation(-1.0,1.0) == -1.0
    assert activations.multiparam_relu_activation(1.0,0.0) == 1.0
    assert activations.multiparam_relu_activation(0.0,0.0) == 0.0
    assert activations.multiparam_relu_activation(-1.0,0.0) == 0.0
    assert activations.multiparam_relu_activation(1.0,-1.0) == 1.0
    assert activations.multiparam_relu_activation(0.0,-1.0) == 0.0
    assert activations.multiparam_relu_activation(-1.0,-1.0) == 1.0

def test_multiparam_elu():
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,-1,-1), -0.026884680254211635)
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,-1.0,0.0), -0.0730801377989552)
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,-1.0,1.0), -0.1986524106001829)
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,0.0,-1.0), -0.0730801377989552)
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,0.0,0.0), -0.1986524106001829)
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,0.0,1.0), -0.5399932379140622)
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,1.0,-1.0), -0.1986524106001829)
    assert_almost_equal(activations.multiparam_elu_activation(-1.0,1.0,0.0), -0.5399932379140622)
    assert activations.multiparam_elu_activation(-1.0,1.0,1.0) == -1.0
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,-1.0,-1.0), -0.024845257339674083)
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,-1.0,0.0), -0.06753641154982476)
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,-1.0,1.0), -0.18358300027522023)
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,0.0,-1.0), -0.06753641154982477)
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,0.0,0.0), -0.18358300027522023)
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,0.0,1.0), -0.49903033366212307)
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,1.0,-1.0), -0.18358300027522026)
    assert_almost_equal(activations.multiparam_elu_activation(-0.5,1.0,0.0), -0.49903033366212307)
    assert activations.multiparam_elu_activation(-0.5,1.0,1.0) == -0.5
    assert activations.multiparam_elu_activation(0.0,-1.0,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,-1.0,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,-1.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,0.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,1.0,-1.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_elu_activation(0.0,1.0,1.0) == 0.0
    assert activations.multiparam_elu_activation(1.0,-1.0,-1.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,-1.0,0.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,-1.0,1.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,0.0,-1.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,0.0,0.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,0.0,1.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,1.0,-1.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,1.0,0.0) == 1.0
    assert activations.multiparam_elu_activation(1.0,1.0,1.0) == 1.0

def test_weighted_lu():
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.0,-1.0), -0.0730801377989552)
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.0,0.0), -0.1986524106001829)
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.0,1.0), -0.5399932379140622)
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.5,-1.0), 0.4634599311005224)
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.5,0.0), -0.09932620530009145)
    assert_almost_equal(activations.weighted_lu_activation(-1.0,0.5,1.0), -0.7699966189570311)
    assert activations.weighted_lu_activation(-1.0,1.0,-1.0) == 1.0
    assert activations.weighted_lu_activation(-1.0,1.0,0.0) == -0.0
    assert activations.weighted_lu_activation(-1.0,1.0,1.0) == -1.0
    assert_almost_equal(activations.weighted_lu_activation(-0.5,0.0,-1.0), -0.06753641154982476)
    assert_almost_equal(activations.weighted_lu_activation(-0.5,0.0,0.0), -0.18358300027522023)
    assert_almost_equal(activations.weighted_lu_activation(-0.5,0.0,1.0), -0.49903033366212307)
    assert_almost_equal(activations.weighted_lu_activation(-0.5,0.5,-1.0), 0.21623179422508762)
    assert_almost_equal(activations.weighted_lu_activation(-0.5,0.5,0.0), -0.09179150013761012)
    assert_almost_equal(activations.weighted_lu_activation(-0.5,0.5,1.0), -0.49951516683106156)
    assert activations.weighted_lu_activation(-0.5,1.0,-1.0) == 0.5
    assert activations.weighted_lu_activation(-0.5,1.0,0.0) == -0.0
    assert activations.weighted_lu_activation(-0.5,1.0,1.0) == -0.5
    assert activations.weighted_lu_activation(0.0,0.0,-1.0) == 0.0
    assert activations.weighted_lu_activation(0.0,1.0,1.0) == 0.0
    assert activations.weighted_lu_activation(1.0,0.0,-1.0) == 1.0
    assert activations.weighted_lu_activation(1.0,1.0,1.0) == 1.0

def test_multiparam_relu_softplus():
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,0.0,0.0), 0.0013430696978235935)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,0.0,0.5), -0.4993284651510882)
    assert activations.multiparam_relu_softplus_activation(-1.0,0.0,1.0) == -1.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,0.5,0.0), 0.5006715348489118)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-1.0,0.5,0.5), 0.0003357674244559017)
    assert activations.multiparam_relu_softplus_activation(-1.0,0.5,1.0) == -0.5
    assert activations.multiparam_relu_softplus_activation(-1.0,1.0,0.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(-1.0,1.0,0.5) == 0.5
    assert activations.multiparam_relu_softplus_activation(-1.0,1.0,1.0) == 0.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,0.0,0.0), 0.015777946858509913)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,0.0,0.5), -0.24211102657074504)
    assert activations.multiparam_relu_softplus_activation(-0.5,0.0,1.0) == -0.5
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,0.5,0.0), 0.25788897342925493)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(-0.5,0.5,0.5), 0.003944486714627465)
    assert activations.multiparam_relu_softplus_activation(-0.5,0.5,1.0) == -0.25
    assert activations.multiparam_relu_softplus_activation(-0.5,1.0,0.0) == 0.5
    assert activations.multiparam_relu_softplus_activation(-0.5,1.0,0.5) == 0.25
    assert activations.multiparam_relu_softplus_activation(-0.5,1.0,1.0) == 0.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.0,0.0), 0.13862943611198905)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.0,0.5), 0.06931471805599453)
    assert activations.multiparam_relu_softplus_activation(0.0,0.0,1.0) == 0.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.5,0.0), 0.06931471805599453)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(0.0,0.5,0.5), 0.03465735902799726)
    assert activations.multiparam_relu_softplus_activation(0.0,0.5,1.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(0.0,1.0,0.0) == 0.0
    assert activations.multiparam_relu_softplus_activation(0.0,1.0,0.5) == 0.0
    assert activations.multiparam_relu_softplus_activation(0.0,1.0,1.0) == 0.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.0,0.0), 1.0013430696978236)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.0,0.5), 1.000671534848912)
    assert activations.multiparam_relu_softplus_activation(1.0,0.0,1.0) == 1.0
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.5,0.0), 1.000671534848912)
    assert_almost_equal(activations.multiparam_relu_softplus_activation(1.0,0.5,0.5), 1.000335767424456)
    assert activations.multiparam_relu_softplus_activation(1.0,0.5,1.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(1.0,1.0,0.0) == 1.0
    assert activations.multiparam_relu_softplus_activation(1.0,1.0,0.5) == 1.0
    assert activations.multiparam_relu_softplus_activation(1.0,1.0,1.0) == 1.0

##    for x in [-1.0,-0.5,0.0,1.0]:
##        for a in [0.0,0.5,1.0]:
##            for b in [0.0,0.5,1.0]:
##                result = activations.multiparam_relu_softplus_activation(x,a,b)
##                if modf(result*100)[0] == 0.0:
##                    print("assert activations.multiparam_relu_softplus_activation({0!r},{1!r},{2!r}) == {3!r}".format(
##                        x, float(a), float(b), result))
##                else:
##                    print("assert_almost_equal(activations.multiparam_relu_softplus_activation({0!r},{1!r},{2!r}), {3!r})".format(
##                        x, float(a), float(b), result))

def test_clamped_tanh_step():
    assert activations.clamped_tanh_step_activation(2.0,1.0) == 1.0 # clamped
    assert activations.clamped_tanh_step_activation(2.0,-1.0) == 1.0 # step
    assert activations.clamped_tanh_step_activation(1.0,1.0) == 1.0 # clamped
    assert activations.clamped_tanh_step_activation(1.0,-1.0) == 1.0 # step
    assert activations.clamped_tanh_step_activation(0.0,1.0) == 0.0 # clamped
    assert activations.clamped_tanh_step_activation(0.0,0.0) == 0.0 # tanh
    assert activations.clamped_tanh_step_activation(-1.0,1.0) == -1.0 # clamped
    assert activations.clamped_tanh_step_activation(-1.0,-1.0) == -1.0 # step
    assert activations.clamped_tanh_step_activation(-2.0,1.0) == -1.0 # clamped
    assert activations.clamped_tanh_step_activation(-2.0,-1.0) == -1.0 # step
    assert activations.clamped_tanh_step_activation(0.5,-1.0) == 1.0 # step
    assert activations.clamped_tanh_step_activation(-0.5,-1.0) == -1.0 # step


def test_multiparam_sigmoid():
    assert activations.multiparam_sigmoid_activation(2.0,1.0) == 1.0
    assert activations.multiparam_sigmoid_activation(2.0,-1.0) == 1.0
    assert activations.multiparam_sigmoid_activation(1.0,1.0) == 1.0
    assert activations.multiparam_sigmoid_activation(1.0,-1.0) == 1.0
    assert activations.multiparam_sigmoid_activation(0.0,1.0) == 0.5
    assert activations.multiparam_sigmoid_activation(0.0,0.0) == 0.5
    assert activations.multiparam_sigmoid_activation(-1.0,1.0) == 0.0
    assert activations.multiparam_sigmoid_activation(-1.0,-1.0) == 0.0
    assert activations.multiparam_sigmoid_activation(-2.0,1.0) == 0.0
    assert activations.multiparam_sigmoid_activation(-2.0,-1.0) == 0.0
    assert activations.multiparam_sigmoid_activation(0.5,-1.0) == 1.0
    assert activations.multiparam_sigmoid_activation(-0.5,-1.0) == 0.0


def test_function_set():
    m = multiparameter.MultiParameterSet('activation')
    s = activations.ActivationFunctionSet(m)
    assert s.get('sigmoid') is not None
    assert s.get('tanh') is not None
    assert s.get('sin') is not None
    assert s.get('gauss') is not None
    assert s.get('relu') is not None
    assert s.get('identity') is not None
    assert s.get('clamped') is not None
    assert s.get('inv') is not None
    assert s.get('log') is not None
    assert s.get('exp') is not None
    assert s.get('abs') is not None
    assert s.get('hat') is not None
    assert s.get('square') is not None
    assert s.get('cube') is not None
    assert m.get_MPF('multiparam_relu', 'activation') is not None
    assert m.get_MPF('multiparam_relu_softplus', 'activation') is not None
    assert m.get_MPF('multiparam_elu', 'activation') is not None
    assert m.get_MPF('weighted_lu', 'activation') is not None
    assert m.get_MPF('clamped_tanh_step', 'activation') is not None
    assert m.get_MPF('multiparam_sigmoid', 'activation') is not None

    assert s.is_valid('sigmoid')
    assert s.is_valid('tanh')
    assert s.is_valid('sin')
    assert s.is_valid('gauss')
    assert s.is_valid('relu')
    assert s.is_valid('identity')
    assert s.is_valid('clamped')
    assert s.is_valid('inv')
    assert s.is_valid('log')
    assert s.is_valid('exp')
    assert s.is_valid('abs')
    assert s.is_valid('hat')
    assert s.is_valid('square')
    assert s.is_valid('cube')
    assert s.is_valid('multiparam_relu')
    assert s.is_valid('multiparam_relu_softplus')
    assert s.is_valid('multiparam_elu')
    assert s.is_valid('weighted_lu')
    assert s.is_valid('clamped_tanh_step')
    assert s.is_valid('multiparam_sigmoid')

    assert not s.is_valid('foo')

def test_bad_add1():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    try:
        config.genome_config.add_activation('1.0',1.0)
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
        config.genome_config.add_activation('dud_function',dud_function)
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

    assert config.genome_config.get_activation_MPF('multiparam_relu') is not None
    assert config.genome_config.get_activation_MPF('multiparam_relu_softplus') is not None
    assert config.genome_config.get_activation_MPF('multiparam_elu') is not None
    assert config.genome_config.get_activation_MPF('weighted_lu') is not None
    assert config.genome_config.get_activation_MPF('clamped_tanh_step') is not None
    assert config.genome_config.get_activation_MPF('multiparam_sigmoid') is not None

    try:
        ignored = config.genome_config.get_activation_MPF('foo')
    except LookupError:
        pass
    else:
        raise Exception("Should have had a LookupError/derived for get_activation_MPF 'foo'")

def test_get_MPF_Instance_simple():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    assert config.genome_config.get_activation_MPF_Instance('multiparam_relu') is not None
    assert config.genome_config.get_activation_MPF_Instance('multiparam_relu_softplus') is not None
    assert config.genome_config.get_activation_MPF_Instance('multiparam_elu') is not None
    assert config.genome_config.get_activation_MPF_Instance('weighted_lu') is not None
    assert config.genome_config.get_activation_MPF_Instance('clamped_tanh_step') is not None
    assert config.genome_config.get_activation_MPF_Instance('multiparam_sigmoid') is not None

    try:
        ignored = config.genome_config.get_activation_MPF_Instance('foo')
    except LookupError:
        pass
    else:
        raise Exception("Should have had a LookupError/derived for get_activation_MPF_Instance 'foo'")

def test_get_MPF_Instance_complex():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    assert config.genome_config.get_activation_MPF_Instance('multiparam_relu(0.5)') is not None
    assert config.genome_config.get_activation_MPF_Instance('multiparam_relu_softplus(0.5,0.5)') is not None
    assert config.genome_config.get_activation_MPF_Instance('multiparam_elu(0.5,0.5)') is not None
    assert config.genome_config.get_activation_MPF_Instance('weighted_lu(0.5,0.5)') is not None
    assert config.genome_config.get_activation_MPF_Instance('clamped_tanh_step(0.5)') is not None
    assert config.genome_config.get_activation_MPF_Instance('multiparam_sigmoid(0.5)') is not None

    try:
        ignored = config.genome_config.get_activation_MPF_Instance('foo(0.5)')
    except LookupError:
        pass
    else:
        raise Exception("Should have had a LookupError/derived for get_activation_MPF_Instance 'foo(0.5)'")

    try:
        ignored = config.genome_config.get_activation_MPF_Instance('multiparam_relu(0.5,0.5,0.5)')
    except RuntimeError:
        pass
    else:
        raise Exception("Should have had a RuntimeError/derived for get_activation_MPF_Instance 'multiparam_relu(0.5,0.5,0.5)'")

if __name__ == '__main__':
    test_sigmoid()
    test_tanh()
    test_sin()
    test_gauss()
    test_relu()
    test_softplus()
    test_identity()
    test_clamped()
    test_inv()
    test_log()
    test_exp()
    test_abs()
    test_hat()
    test_square()
    test_cube()
    test_multiparam_relu()
    test_multiparam_elu()
    test_weighted_lu()
    test_multiparam_relu_softplus()
    test_clamped_tanh_step()
    test_multiparam_sigmoid()
    test_function_set()
    test_get_MPF()
    test_get_MPF_Instance_simple()
    test_get_MPF_Instance_complex()
    test_bad_add1()
    test_bad_add2()
