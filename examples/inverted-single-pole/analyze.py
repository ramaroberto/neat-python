#!/usr/bin/python2
"""

This file takes the pickle files dropped in the data directory and analyzes
the model in them. The model has the samples, observations and distance
(NEAT distance) already set at the time of saving, so it can be used as mere
container of data to try other models.

model.samples = samples
model.observations = real fitness of each sample (present also in model.samples[k].real_fitness)
model.kf.df = distance function, it takes 2 genomes and returns NEAT distance.

"""

from copy import deepcopy
import dill
import pickle
import numpy as np
import random
from subprocess import Popen, PIPE
# import GPy

import os
from os import listdir
from os.path import isfile, join

# Import the local version of neat instead of the one in the library
import imp, sys
from os import path
local_neat_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
f, pathname, desc = imp.find_module("neat", [local_neat_path])
local_neat = imp.load_module("neat", f, pathname, desc)

import neat
from neat.models.gp import RBF, GaussianProcessModel

random.seed(1)

DIRECTORY = "data"

def load_data_file_est(path):
    with open(path, "rb") as input:
        estimations = pickle.load(input)
    return estimations

def load_data_file(path):
    with open(path, "rb") as input:
        gpm = pickle.load(input)
        
        kernel = RBF(length=1, sigma=1, noise=1)
        kernel.df = gpm.kf.df
        kernel.ls = gpm.kf.ls
        kernel.ss = gpm.kf.ss
        kernel.noise = gpm.kf.noise
        
        new_gpm = GaussianProcessModel(kernel, 1, 1, optimize_noise=True)
        new_gpm.samples = gpm.samples
        new_gpm.observations = gpm.observations
        new_gpm.kernel = gpm.kernel
        new_gpm.observations_mean = gpm.observations_mean
        new_gpm.zero_meaned_obs = gpm.zero_meaned_obs
        new_gpm.matrixL = gpm.matrixL
        new_gpm.alpha = gpm.alpha
        new_gpm.K_inv = gpm.K_inv
        new_gpm.gram = gpm.gram
        new_gpm.optimize_noise = gpm.optimize_noise
        
        return new_gpm

def get_average_sample_error(gpm):
    avg_std_samples = 0
    avg_samples_error = 0
    for i, sample in enumerate(gpm.samples):
        mean, std = gpm.predict(sample)
        error = abs(mean - gpm.observations[i,0])
        avg_samples_error += error
        avg_std_samples += std
        
        # if 0.1 < abs(gpm.observations[i,0] - mean):
        #     print "Warning: ", gpm.observations[i,0], gpm.zero_meaned_obs[i,0], mean, std
        #     print np.sqrt(gpm.gram[i,:])
        #     print np.sqrt(gpm.kernel[i,:])
        #     print np.sqrt(gpm.kernel[i,i])
        
        avg_samples_error /= float(len(gpm.samples))
        avg_std_samples /= float(len(gpm.samples))
    return avg_samples_error, avg_std_samples

def print_m(mat):
    for i in range(mat.shape[0]):
        print np.sqrt(mat[i])
        
def mat_to_text(mat):
    n, m = mat.shape
    output = "["
    for i in range(n):
        if i == 0:
            output += "["
        else:
            output += ", \n ["
        for j in range(m):
            if j != 0:
                output += ", "
            output += str(mat[i, j])
        output += "]"
    output += "]"
    
    return output
            

def cbcopy(text):
    p = Popen(['xsel','-b'], stdin=PIPE)
    p.communicate(input=text)

def analyze(gpm, i=None, estimations=None):
    np.set_printoptions(precision=4)
    lgram = np.sqrt(gpm.gram).flatten().tolist()
    max_gram = max(filter(lambda e: e > 0, lgram))
    min_gram = min(filter(lambda e: e > 0, lgram))
    
    samples = sorted(map(lambda g: (g.key, g.fitness, g.real_fitness), gpm.samples), key=lambda g: g[0])
    print "\n\n\nGeneration " + str(i) + ":"
    print "Error on Samples:", get_average_sample_error(gpm), "("+str(len(gpm.samples))+")"
    print "HPS:", gpm.kf.get_hps()
    print "Warnings Count:", gpm.negative_variance_warning_count
    print "Max/Min GRAM:", max_gram, min_gram
    print "Mean/std Obs:", np.mean(gpm.observations), np.std(gpm.observations)
    print "Mean/std GRAM:", np.mean(lgram), np.std(lgram)
    print "Rank/Size/Gram:", np.linalg.matrix_rank(gpm.gram), gpm.gram.shape[0], np.sqrt(gpm.gram[0])
    cbcopy(mat_to_text(gpm.gram))
    print "Rank/Size/Kernel:", np.linalg.matrix_rank(gpm.kernel), gpm.kernel.shape[0], gpm.kernel[0]
    # print_m(gpm.kernel)
    print "Likelihood:", gpm.get_log_likelihood()
    print "Samples:", len(samples), samples
    print "Observations:", len(gpm.observations), gpm.observations.transpose()
    
    if estimations:
        avg_error = 0
        count = 0
        for estimation in estimations:
            fitness = estimation[0]
            genome = estimation[1]
            
            if fitness < 0.95:
                continue
            
            count += 1
            mean, std = gpm.predict(genome)
            error = abs(mean - fitness) - std
            
            if error < 0:
                error = 0
            
            if genome.key in map(lambda g: g.key, gpm.samples):
                print genome.key
                print map(lambda g: (g[1].key, gpm.predict(g[1])[0], gpm.observations[g[0],0]), list(enumerate(gpm.samples)))
                print "->",
                for i, g in enumerate(gpm.samples):
                    if genome.key == g.key:
                        print gpm.observations[i],
            print fitness, mean+std, mean, std, error
            
            avg_error += error
            
        avg_error /= float(count)
        print "Estimations error:", avg_error, "("+str(count)+")"

def analyze_one(filename):
    model = load_data_file(DIRECTORY + "/" + filename)
    # print(len(model.samples), len(model.observations))
    # model.kf.set_hps(*params)
    model.compute(model.samples, model.observations)
    model.optimize(quiet=True, bounded=True, fevals=200)
    # print(model.samples[36])
    # print(model.samples[34])
    # print(model.kf.df(model.samples[36], model.samples[34]))
    analyze(model, 0)
    
def analyze_all():
    data_files = sorted([f for f in listdir(DIRECTORY) if isfile(join(DIRECTORY, f))])
    data_files_model = filter(lambda s: s.split("_")[1] == "model.pkl", data_files)

    models = []
    for data_file in data_files_model:
        models.append(load_data_file(DIRECTORY + "/" + data_file))
    
    for i, model in enumerate(models):
        analyze(model, i)
        model.optimize(quiet=True, bounded=True, fevals=200)
        # analyze(model, i)

def main():
    analyze_one("64_model.pkl")

if __name__ == '__main__':
    main()