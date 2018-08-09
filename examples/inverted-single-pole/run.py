#!/usr/bin/python2

import logging
import pickle
import gym
import math
import numpy as np
import time

from cartpole import CartPoleEnv
from copy import deepcopy

import sys
from os import path

# Import the local version of neat instead of the one in the library
import imp, sys
from os import path
local_neat_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
f, pathname, desc = imp.find_module("neat", [local_neat_path])
local_neat = imp.load_module("neat", f, pathname, desc)

import neat

# Config for FeedForwardNetwork.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_inverted_single_pole')

num_workers = 6
env = CartPoleEnv()
envs_pool = [CartPoleEnv() for i in range(num_workers*2)]

for i, lenv in enumerate(envs_pool):
    lenv.id = i
                            
# Initialize population attaching statistics reporter.
def ini_pop(state, stats, config, output):
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop

def eval_genome(genome, config, visualize=False):
    
    # Get a cartpole's environment from the pool
    local_env = envs_pool.pop(0)
    
    rewards = []
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for i in xrange(1):
        obs = local_env.reset()
        reward_streak = 0
        for j in xrange(200): # 5 seconds of total time: 5 / 0.025 (ts) = 200
            if visualize:
                local_env.render()
            # +/-3, +/-5?, +/-2pi, +/-2pi
            x, x_dot, theta, theta_dot = obs
            out = net.activate([x/4.0, x_dot/10.0, math.cos(theta), math.sin(theta), theta_dot/math.pi])
            force = max(-10, min(10, out[0] * 20 - 10))
            obs, reward, done, info = local_env.step(force)
            reward_streak += reward
            if done:
                break
            if abs(reward) < 1.0:
                rewards.append(reward_streak)
                reward_streak = 0
    
    # Return the environment to the pool after use
    envs_pool.append(local_env)
    
    rewards.append(reward_streak)
    return max(rewards)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run_neat(gens, env, config, max_trials=100, output=True):
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    winner = pop.run(pe.evaluate, gens)
    
    ran_generations = len(stats.generation_statistics)-1
    
    return winner, stats, ran_generations
    
def run_experiment():
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # env = gym.make("CartPole-v1")

    # Run!
    results = []
    n_experiments = 1
    for i in xrange(n_experiments):
        winner, stats, generations = run_neat(500, env, config)
        results.append((winner, stats, generations))
        print "Experiment " + (str(i+1)) + "/" + str(n_experiments) + " finished."
        time.sleep(2)
    
    # Save results to file
    if n_experiments > 1:
        f = open("experiment.data", "wb")
        pickle.dump(results, f)
    else:
        # Visualize
        reward = eval_genome(winner, config, visualize=True)
        print "Total reward is:", reward, eval_genome(winner, config)

def load_experiment():
    # Load results from file
    f = open("experiment.data", "rb")
    results = pickle.load(f)
    
    # Get a result
    winner, stats, generations =  results[0]
    
    # Visualize
    reward = eval_genome(winner, config, visualize=True)
    print "Total reward is:", reward, eval_genome(winner, config)

# If run as script.
if __name__ == '__main__':
    
    run_experiment()
    
    # load_experiment()
    
    # Save net if wished reused and draw it + winner to file.
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # draw_net(winner_net, filename="neat_pole_balancing_winner")
    # with open('neat_pole_balancing_winner.pkl', 'wb') as output:
    #     pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)

