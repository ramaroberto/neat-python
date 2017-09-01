from __future__ import print_function, division

#import math
import os
import warnings

warnings.simplefilter('default')

import neat

#MAX_MUTATE = math.ceil(10000/0.01)
MAX_MUTATE = (10000/0.01)

##from neat.six_util import iterkeys

def test_float_attribute():
    """Make sure can alter without affecting others"""
    gid = 42
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration_2outputs')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    #config.genome_config.num_outputs = 2

    g = neat.DefaultGenome(gid)
    g.configure_new(config.genome_config)

    bias0 = getattr(g.nodes[0], 'bias')
    bias1 = getattr(g.nodes[1], 'bias')

    count = 0

    while bias0 == getattr(g.nodes[0], 'bias'):
        if count > MAX_MUTATE:
            raise RuntimeError("Tried mutating {0:n} times without success".format(count))
        g.nodes[0].mutate(config.genome_config)
        count += 1

    assert bias1 == getattr(g.nodes[1], 'bias'), "Bias1 changed from {0:n} to {1:n}".format(
        bias1, getattr(g.nodes[1], 'bias'))

    bias0 = getattr(g.nodes[0], 'bias')

    ignored_net = neat.nn.FeedForwardNetwork.create(g, config)

    count = 0

    while bias0 == getattr(g.nodes[0], 'bias'):
        if count > MAX_MUTATE:
            raise RuntimeError("Tried mutating {0:n} times without success".format(count))
        g.nodes[0].mutate(config.genome_config)
        count += 1

    assert bias1 == getattr(g.nodes[1], 'bias'), "Bias1 changed from {0:n} to {1:n}".format(
        bias1, getattr(g.nodes[1], 'bias'))

def test_mixed_float_init():
    """Make sure having uniform for multiparam does not alter (default) for bias, response, etc"""
    gid = 42
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration6')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    g = neat.DefaultGenome(gid)
    g.configure_new(config.genome_config)

    if config.genome_config.weight_init_type.lower() not in ('gaussian', 'normal'):
        raise AssertionError(
            "Weight_init_type of genome_config is {0!r} not gaussian/normal".format(
                config.genome_config.weight_init_type))

    ignored_net = neat.nn.FeedForwardNetwork.create(g, config)

    if config.genome_config.weight_init_type.lower() not in ('gaussian', 'normal'):
        raise AssertionError(
            "Weight_init_type of genome_config is {0!r} not gaussian/normal".format(
                config.genome_config.weight_init_type))

def test_attribute_repro(config_file='test_configuration_2outputs'):
    """Make sure that not tied together after reproduction, even if clone"""

    gid = 42
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    g1 = neat.DefaultGenome(gid)
    g1.configure_new(config.genome_config)
    g1.fitness = 1.0

    ignored_net = neat.nn.FeedForwardNetwork.create(g1, config)

    g2 = neat.DefaultGenome(gid+1)
    g2.configure_crossover(g1, g1, config.genome_config)

    ignored_net = neat.nn.FeedForwardNetwork.create(g2, config)

    bias0_g1 = getattr(g1.nodes[0], 'bias')
    bias0_g2 = getattr(g2.nodes[0], 'bias')

    assert bias0_g1 == bias0_g2, "Clone has bias0 {0:n} but original has bias0 {1:n}".format(
        bias0_g2, bias0_g1)

    count = 0

    while bias0_g1 == getattr(g1.nodes[0], 'bias'):
        if count > MAX_MUTATE:
            raise RuntimeError("Tried mutating {0:n} times without success".format(count))
        g1.nodes[0].mutate(config.genome_config)
        count += 1

    assert bias0_g2 == getattr(g2.nodes[0], 'bias'), "Bias0 of clone changed from {0:n} to {1:n} (original {2:n})".format(
        bias0_g2, getattr(g2.nodes[0], 'bias'), getattr(g1.nodes[0], 'bias'))

def test_attribute_repro_config6():
    """Repeat test with having multiparam funcs in genome"""
    test_attribute_repro(config_file='test_configuration6')

def test_multiparam_repro(config_file='test_configuration6'):
    """Make sure that not tied together after reproduction, even if clone"""

    gid = 42
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    g1 = neat.DefaultGenome(gid)
    g1.configure_new(config.genome_config)
    g1.fitness = 1.0

    a1_dict = g1.nodes[0].activation.get_values(None)

    ignored_net = neat.nn.FeedForwardNetwork.create(g1, config)

    g2 = neat.DefaultGenome(gid+1)
    g2.configure_crossover(g1, g1, config.genome_config)

    a2_dict = g2.nodes[0].activation.get_values(None)

    ignored_net = neat.nn.FeedForwardNetwork.create(g2, config)

    a1 = g1.nodes[0].activation.get_values('tilt')
    a2 = g2.nodes[0].activation.get_values('tilt')

    assert a1 == a2, "Clone has 'tilt' {0:n} but original has 'tilt' {1:n}".format(
        a1, a2)

    assert a1_dict['tilt'] == a2_dict['tilt'], "Clone has {0!r} but original has {1!r}".format(
        a1_dict, a2_dict)

    print("g1: {!r} at {!s}".format(g1.nodes[0].activation,id(g1.nodes[0].activation)))
    print("g2: {!r} at {!s}".format(g2.nodes[0].activation,id(g2.nodes[0].activation)))

    count = 0

    while a1 == g1.nodes[0].activation.get_values('tilt'):
        if count > MAX_MUTATE:
            raise RuntimeError("Tried mutating {0:n} times without success".format(count))
        g1.nodes[0].mutate(config.genome_config)
        count += 1

##    print("g1: {!r} at {!s}".format(g1.nodes[0].activation,id(g1.nodes[0].activation)))
##    print("g2: {!r} at {!s}".format(g2.nodes[0].activation,id(g2.nodes[0].activation)))

    assert a2 == g2.nodes[0].activation.get_values('tilt'), "Clone 'tilt' changed from {0:n} to {1:n} (original {2:n})".format(
        a2, g2.nodes[0].activation.get_values('tilt'), g1.nodes[0].activation.get_values('tilt'))


if __name__ == '__main__':
    test_float_attribute()
    test_mixed_float_init()
    test_attribute_repro()
    test_attribute_repro_config6()
    test_multiparam_repro()
