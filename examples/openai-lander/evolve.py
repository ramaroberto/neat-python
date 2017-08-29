"""
Evolve a control/reward estimation network for the OpenAI Gym
LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg
"""
from __future__ import print_function, division

import multiprocessing
import os
import pickle
import random
import time

import gym
import gym.wrappers

import matplotlib.pyplot as plt
import numpy as np

import neat
import visualize

NUM_CORES = os.cpu_count()
if NUM_CORES is None:
    NUM_CORES = 4

orig_env = gym.make('LunarLander-v2')

print("action space: {0!r}".format(orig_env.action_space))
print("observation space: {0!r}".format(orig_env.observation_space))

env = gym.wrappers.Monitor(orig_env, 'results', force=True)


class LanderGenome(neat.DefaultGenome):
    """A DefaultGenome with the addition of a `discount` evolved parameter."""
    def __init__(self, key):
        super(LanderGenome, self).__init__(key)
        self.discount = None

    def configure_new(self, config):
        super(LanderGenome, self).configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super(LanderGenome, self).configure_crossover(genome1, genome2, config)
        if genome1.fitness > genome2.fitness:
            self.discount = genome1.discount
        elif genome1.fitness < genome2.fitness:
            self.discount = genome2.discount
        else:
            self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super(LanderGenome, self).mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super(LanderGenome, self).distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff

    def __str__(self):
        return "Reward discount: {0:n}\n{1!s}".format(self.discount,
                                                  super(LanderGenome,self).__str__())


def compute_fitness(genome, net, episodes, min_reward, max_reward):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for ignored_score, data in episodes:
        # Compute normalized discounted reward.
        dr = np.convolve(data[:,-1], discount_function)[m:]
        dr = 2 * (dr - min_reward) / (max_reward - min_reward) - 1.0
        dr = np.clip(dr, -1.0, 1.0)

        for row, dr in zip(data, dr):
            observation = row[:8]
            action = int(row[8])
            output = net.activate(observation)
            reward_error.append(float((output[action] - dr) ** 2))

    return reward_error


class PooledErrorCompute(object):
    def __init__(self, control_random=False):
        if NUM_CORES < 2:
            self.pool = None
        else:
            self.pool = multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0
        self.control_random = control_random

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    def simulate(self, nets):
        scores = []
        use_seed = env.seed()
        for ignored_genome, net in nets:
            if self.control_random:
                env.seed(seed=use_seed)
            observation = env.reset()
            step = 0
            data = []
            while 1:
                step += 1
                if step < 200 and random.random() < ((200-step)/500):
                    action = env.action_space.sample()
                else:
                    output = net.activate(observation)
                    action = np.argmax(output)

                observation, reward, done, ignored_info = env.step(action)
                data.append(np.hstack((observation, action, reward)))

                if done:
                    break

            data = np.array(data)
            score = np.sum(data[:,-1])
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)

            self.test_episodes.append((score, data))

        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        nets = []
        for ignored_gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        print("network creation time {0:n}".format(time.time() - t0))
        t0 = time.time()

        # Periodically generate a new set of episodes for comparison.
        if (self.generation % 10) == 1:
            self.test_episodes = self.test_episodes[-300:]
            self.simulate(nets)
            print("simulation run time {0:n}".format(time.time() - t0))
            t0 = time.time()

        # Assign a composite fitness to each genome; genomes can make
        # progress either by improving their total reward or by making
        # more accurate reward estimates.
        # TODO: I don't think the above comment is what is happening...
        print("Evaluating {0:n} test episodes".format(len(self.test_episodes)))
        if self.pool is None:
            for genome, net in nets:
                reward_error = compute_fitness(genome,
                                               net,
                                               self.test_episodes,
                                               self.min_reward,
                                               self.max_reward)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        else:
            jobs = []
            for genome, net in nets:
                jobs.append(self.pool.apply_async(compute_fitness,
                    (genome, net,
                     self.test_episodes, self.min_reward, self.max_reward)))

            for job, (ignored_genome_id, genome) in zip(jobs, genomes):
                reward_error = job.get(timeout=None)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)

        print("final fitness compute time {0:n}\n".format(time.time() - t0))


def run(control_random=False,filename_ext="svg"):
    """Main loop."""
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(LanderGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute(control_random=control_random)
    while 1:
        try:
            ignored_gen_best = pop.run(ec.evaluate_genomes, 5)

            #print(gen_best)

            visualize.plot_stats(stats,
                                 ylog=False,
                                 view=False,
                                 filename="fitness.{0!s}".format(filename_ext))

            plt.plot(ec.episode_score, 'g-', label='score')
            plt.plot(ec.episode_length, 'b-', label='length')
            plt.grid()
            plt.legend(loc='best')
            plt.savefig("scores.{0!s}".format(filename_ext))
            plt.close()

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0:n}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0:n}".format(mfs))

            # Use the best genomes seen so far as an ensemble-ish control system.
            best_genomes = stats.best_unique_genomes(3)
            best_networks = []
            for g in best_genomes:
                best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))

            solved = True
            best_scores = []
            for k in range(100):
                observation = env.reset()
                score = 0
                step = 0
                while 1:
                    step += 1
                    # Use the total reward estimates from all three networks to
                    # determine the best action given the current state.
                    # TODO: Use softmax on outputs and add up to determine action
                    votes = np.zeros((4,))
                    for n in best_networks:
                        output = n.activate(observation)
                        votes[np.argmax(output)] += 1

                    best_action = np.argmax(votes)
                    observation, reward, done, ignored_info = env.step(best_action)
                    score += reward
                    env.render()
                    if done:
                        break

                ec.episode_score.append(score)
                ec.episode_length.append(step)

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                print(k, score, avg_score)
                if avg_score < 200:
                    solved = False
                    break

            if solved:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name+'.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False,
                                       filename=name+"-net.gv")
                    visualize.draw_net(config, g, view=False,
                                       filename=name+"-net-enabled.gv",
                                       show_disabled=False)
                    visualize.draw_net(config, g, view=False,
                                       filename=name+"-net-enabled-pruned.gv",
                                       show_disabled=False, prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()


if __name__ == '__main__':
    run()
