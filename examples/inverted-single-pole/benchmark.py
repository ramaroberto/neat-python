#!/usr/bin/python2

import pickle

# Import the local version of neat instead of the one in the library
import imp, sys
from os import path
local_neat_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
f, pathname, desc = imp.find_module("neat", [local_neat_path])
local_neat = imp.load_module("neat", f, pathname, desc)

import neat
import numpy as np
import random

def get_binned_genomes(gens_species, bins_limits):
    bins = [[] for i in range(len(bins_limits))]
    for gs in gens_species:
        for s in gs.species.values():
            for genome in s.members.values():
                # Fill the bins
                for i in range(len(bins_limits)-1):
                    if bins_limits[i] < genome.fitness and \
                        genome.fitness < bins_limits[i+1]:
                        bins[i].append(genome)
                if bins_limits[-1] < genome.fitness:
                    bins[-1].append(genome)
    return bins

def get_samples(bins, sample_size, proportions=[.1, .2, .4, .2, .1]):
    # TODO: Check that proportions has uneven elements.
    # Sliding window sampling
    window_side_size = (len(proportions)-1)/2
    samples = []
    for c in range(len(bins)):
        window_lower_bound = max(0, window_side_size - c)
        window_upper_bound = min(window_side_size, len(bins) - c - 1)
        
        # Re-escale proportions
        local_proportions = \
            proportions[window_lower_bound:window_side_size+1] + \
            proportions[window_side_size+1:window_side_size+1 + window_upper_bound]
        
        # Do initial distribution
        weighted_bins = []
        for j, i in enumerate(range(max(0, c - window_side_size - window_lower_bound), c + window_upper_bound + 1)):
            weighted_bins.append((i, local_proportions[j]))
        weighted_bins.sort(key=lambda s: s[1], reverse=True)
        
        leftover = sample_size
        total = sum(local_proportions)
        selections = []
        for i, ratio in weighted_bins:
            number_to_choose = min(len(bins[i]), int((ratio/total) * leftover))
            selections.append([i, number_to_choose])
            total -= ratio
            leftover -= number_to_choose
        
        # Accomodate leftover
        while leftover > 0:
            all_used = True
            for selection in selections:
                if leftover > 0 and selection[1] < len(bins[selection[0]]):
                    all_used = False
                    selection[1] += 1
                    leftover -= 1
            if all_used or leftover <= 0:
                break
        
        # Do the sampling
        sample = []
        for i, number_to_choose in selections:
            choices = random.sample(range(len(bins[i])), number_to_choose)
            sample += map(lambda c: bins[i][c], choices)        
            print "bins["+str(i)+"]("+str(len(choices))+"),",
        print leftover
        
        samples.append(sample)
    return samples
    
def get_percentiles(data):
    return (np.percentile(data, 10),\
        np.percentile(data, 25),\
        np.percentile(data, 50),\
        np.percentile(data, 75),\
        np.percentile(data, 90))

def get_average_sample_error(gpm, testing=None, testing_observations=None, return_percentiles=False):
    errors = []
    if not testing:
        testing = gpm.samples
        testing_observations = np.asarray(gpm.observations).reshape(-1)
    for i, sample in enumerate(testing):
        mean, std = gpm.predict(sample)
        if type(mean) is np.matrix:
            mean = mean[0,0]
        if type(std) is np.matrix:
            std = std[0,0]
        errors.append(abs(mean - testing_observations[i]))
    if return_percentiles:
        return get_percentiles(errors)
    else:
        return np.mean(errors), np.std(errors), max(errors)
    

def analyze(gpm, i=None, testing=None, testing_observations=None):
    np.set_printoptions(precision=4)
    lgram = np.sqrt(gpm.gram).flatten().tolist()
    max_gram = max(filter(lambda e: e > 0, lgram))
    min_gram = min(filter(lambda e: e > 0, lgram))
    
    samples = sorted(map(lambda g: (g.key, g.fitness, g.real_fitness), gpm.samples), key=lambda g: g[0])
    print "\n\n\nGeneration " + str(i) + ":"
    print "Error on Samples (Percentiles):", get_average_sample_error(gpm, return_percentiles=True), "("+str(len(gpm.samples))+")"
    print "Error on Testing (Percentiles):", get_average_sample_error(gpm, testing, testing_observations, return_percentiles=True), "("+str(len(testing))+")"
    print "Error on Samples (Averages):", get_average_sample_error(gpm), "("+str(len(gpm.samples))+")"
    print "Error on Testing (Averages):", get_average_sample_error(gpm, testing, testing_observations), "("+str(len(testing))+")"
    print "HPS:", gpm.kf.get_hps()
    print "Warnings Count:", gpm.negative_variance_warning_count
    print "PD's Error Count:", gpm.hp_not_positive_definite_error_count
    # print "Max/Min GRAM:", max_gram, min_gram
    # print "Mean/std Obs:", np.mean(gpm.observations), np.std(gpm.observations)
    # print "Mean/std GRAM:", np.mean(lgram), np.std(lgram)
    print "Rank/Size/Gram:", np.linalg.matrix_rank(gpm.gram), gpm.gram.shape[0]#, np.sqrt(gpm.gram[0])
    print "Rank/Size/Kernel:", np.linalg.matrix_rank(gpm.kernel), gpm.kernel.shape[0]#, gpm.kernel[0]
    # print "Likelihood:", gpm.get_log_likelihood()
    # print "Samples:", len(samples), samples
    # print "Observations:", len(gpm.observations), gpm.observations.transpose()
    
    return get_average_sample_error(gpm, return_percentiles=True)[2], \
        get_average_sample_error(gpm, testing, testing_observations, return_percentiles=True)[2]

def main(data_filename):
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                "config_inverted_single_pole", neat.surrogate.DefaultSurrogateModel)
    
    # Load data file
    with open(data_filename, "rb") as input:
        gens_species = pickle.load(input)

    sample_size = 64
    bins_limits = [10.0 * i for i in range(0, 10)]
    bins = get_binned_genomes(gens_species, bins_limits)
    samples = get_samples(bins, sample_size, [.33, .34, .33])
    
    medians_training = []
    medians_testing = []
    df = lambda g1, g2: g1.distance(g2, config.genome_config)
    samples_cut = sample_size/2
    for i, sample in enumerate(samples):
    # for i, sample in [(6, samples[6])]:
        training = sample[:samples_cut]
        testing = sample[samples_cut:]
        
        # filtered_training = []
        # for g1 in training:
        #     too_close = False
        #     for g2 in filtered_training:
        #         if df(g1, g2) < 0.1:
        #             too_close = True
        #             break
        #     if not too_close:
        #         filtered_training.append(g1)
        
        surrogate = neat.surrogate.GaussianProcessSurrogateModel(df)
        surrogate.model.kf.set_hps(0.00037, 1., 0.09923)
        surrogate.train(training, map(lambda g: g.real_fitness, training), optimize=True)
        # surrogate.train(training, map(lambda g: g.real_fitness, training), optimize=True)
        
        median_training, median_testing = \
            analyze(surrogate.model, i+1, testing, map(lambda g: g.real_fitness, testing))
        medians_training.append(median_training)
        medians_testing.append(median_testing)
        # predictions = surrogate.predict(sample, None)
        # print predictions
    
    print "\nTraining's medians percentiles: ", get_percentiles(medians_training)
    print "Testing's medians percentiles:  ", get_percentiles(medians_testing)
        
    
if __name__ == '__main__':
    main("data/20181106180100_species.pkl")