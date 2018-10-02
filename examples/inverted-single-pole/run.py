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
    
    # env = gym.make("CartPole-v1")
    
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
        
    
    # Save results to file
    if repetitions > 1:
        f = open(timestamp+"_"+config_name+".data", "wb")
        pickle.dump(results, f)
        return [np.percentile(experiments_generations, 10),\
            np.percentile(experiments_generations, 25),\
            np.percentile(experiments_generations, 50),\
            np.percentile(experiments_generations, 75),\
            np.percentile(experiments_generations, 90)]
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
    reward = eval_genome(winner, config, visualize=False)
    print "Total reward is:", reward, eval_genome(winner, config)

from random import shuffle

def load_manual_nn():
    # bias, x, cos, sin, x_dot, theta_dot
    input_to_output_conns_weights = [-1.27081788905737, -2.16328518046126, -1.17322972615653,  2.05416948816017, -4.17464738664070, -1.92563440012328]
    
    genome = neat.DefaultGenome(0)
    genome.configure_new(config.genome_config)
    connections = sorted(genome.connections.values(), key=lambda conn: conn.key[0])
    
    # Set the weights of the out connections
    for i, ck in enumerate(sorted(genome.connections.keys(), key=lambda k: -k[0])):
        genome.connections[ck].set("weight", input_to_output_conns_weights[i])
    reward = eval_genome(genome, config)
    
    # Add the new node
    genome.add_node(config.genome_config, connections[-1])
    connections[-1].enabled = True # Re-enable the connection
    
    # Define the 1 to out connection's weight
    genome.connections[(1, 0)].weight = -0.21993945865389
    
    # Add the remaining connections
    genome.connections[(-1, 1)].weight = 0.818966241451037 # bias to 1
    genome.add_connection(config.genome_config, -2, 1, 4.45185524142824, True) # x to 1
    genome.add_connection(config.genome_config, -4, 1, -2.06666066002367, True) # sin to 1
    genome.add_connection(config.genome_config, -6, 1, 0.180692888372585, True) # theta_dot to 1
    
    print genome
    print eval_genome(genome, config, visualize=True)
    # print "Total reward is:", reward, eval_genome(genome, config)

# If run as script.
if __name__ == '__main__':
    
    # load_manual_nn()
    # run_experiment("config_inverted_single_pole", repetitions=50, max_generations=500)
    
    # run_experiment("config_inverted_single_pole_stag_20_st_03_ctm", repetitions=200, max_generations=500)
    run_experiment("config_inverted_single_pole_stag_20_st_03", repetitions=200, max_generations=500)
    run_experiment("config_inverted_single_pole_stag_20_st_03_wmp", repetitions=200, max_generations=500)
    
    # run_experiment("config_inverted_single_pole_stag_20_st_03_atfb", repetitions=100, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_20_st_03_wmp", repetitions=100, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_20_st_03_cp_05", repetitions=100, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_20_st_03_tn_6", repetitions=100, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_20_st_03_structural_probs_reduced", repetitions=100, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_20_st_03_no_elitism", repetitions=100, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_15_wmp_05", repetitions=100, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_15", repetitions=50, max_generations=500)
    # run_experiment("config_inverted_single_pole_stag_15_structural_mut_changed", repetitions=50, max_generations=500)
    
    # load_experiment()
    
    # Save net if wished reused and draw it + winner to file.
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # draw_net(winner_net, filename="neat_pole_balancing_winner")
    # with open('neat_pole_balancing_winner.pkl', 'wb') as output:
    #     pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)

# 0.818966241451037    -1.27081788905737    bias
# 4.45185524142824	   -2.16328518046126    x
# 0	                   -1.17322972615653    cos
# -2.06666066002367	    2.05416948816017    sin
# 0	                   -4.17464738664070    x_dot
# 0.180692888372585	   -1.92563440012328    theta_dot
# 0	                   -0.21993945865389    node 7 to out
# 0	                    0