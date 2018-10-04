#!/usr/bin/python2

import logging
import pickle
import gym
import math
import numpy as np
import time
import datetime

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

def eval_genome(genome, config, visualize=False, normalize_input=False):
    
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
                if j == 0:
                    time.sleep(1.0)
                else:
                    time.sleep(0.015)
            
            out = None
            x, x_dot, theta, theta_dot = obs
            if normalize_input:
                # +/-3, +/-10, +/-2pi, +/-2pi
                inp = [1, x/3.0, math.cos(theta), math.sin(theta), x_dot/10.0, theta_dot/2*math.pi]
            else:
                # out = net.activate([1, x, x_dot, math.cos(theta), math.sin(theta), theta_dot])
                # out = net.activate([1, x, math.cos(theta), math.sin(theta), x_dot, theta_dot])
                inp = [1, x, math.cos(theta), math.sin(theta), x_dot, theta_dot]
            
            inp.reverse()    
            out = net.activate(inp)
            force = out[0] * 10
            
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
        genome.fitness = eval_genome(genome, config, normalize_input=True)

def run_neat(gens, env, config, output=True):
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(output))

    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    winner = pop.run(pe.evaluate, gens)
    
    ran_generations = len(stats.generation_statistics) - 1
    
    return winner, stats, ran_generations
    
def run_experiment(config_name, repetitions=1, max_generations=200):
    # Config for FeedForwardNetwork.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                config_name)
                                
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)        
    
    # Save results to file
    if repetitions > 1:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        f = open(timestamp+"_"+config_name+".log", "wa", 0) # buffer size = 0

        # Run!
        results = []
        experiments_generations = []
        for i in xrange(repetitions):
            winner, stats, generations = run_neat(max_generations, env, config)
            results.append((winner, stats, generations))
            experiments_generations.append(generations)
            msg = "Experiment " + (str(i+1)) + "/" + str(repetitions) + " finished.\n"
            msg += "Generations: " + str(generations) + "\n"
            msg += "Generations percentiles: " + \
                str(np.percentile(experiments_generations, 10)) + " " + \
                str(np.percentile(experiments_generations, 25)) + " " + \
                str(np.percentile(experiments_generations, 50)) + " " + \
                str(np.percentile(experiments_generations, 75)) + " " + \
                str(np.percentile(experiments_generations, 90)) + "\n"
            msg += "Generations average/std: " + \
                str(np.mean(experiments_generations)) + " " + \
                str(np.std(experiments_generations)) + "\n\n"

            f.write(msg)
            f.flush()
            print msg
        
        f.close()
        
        f = open(timestamp+"_"+config_name+".data", "wb")
        pickle.dump(results, f)
        return [np.percentile(experiments_generations, 10),\
            np.percentile(experiments_generations, 25),\
            np.percentile(experiments_generations, 50),\
            np.percentile(experiments_generations, 75),\
            np.percentile(experiments_generations, 90)]
    else:
        winner, stats, generations = run_neat(max_generations, env, config)
        reward = eval_genome(winner, config, visualize=True)
        print "Total reward is:", reward, eval_genome(winner, config)

def load_experiment():
    # Load results from file
    f = open("experiment.data", "rb")
    results = pickle.load(f)
    
    # Get a result
    winner, stats, generations =  results[0]
    
    # Visualize
    reward = eval_genome(winner, config, visualize=False)
    print "Total reward is:", reward, eval_genome(winner, config)

# If run as script.
if __name__ == '__main__':
    run_experiment("config_inverted_single_pole_stag", repetitions=200, max_generations=500)
    # load_experiment()