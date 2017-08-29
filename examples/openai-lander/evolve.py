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
    discount_use_reward = False
    simulator_instance = None
    """
    A DefaultGenome with the addition of a `discount` evolved parameter
    and optional total reward tracking.
    """
    def __init__(self, key):
        super(LanderGenome, self).__init__(key)
        self.discount = None
        self.reward = None

    @classmethod
    def set_discount_use_reward(cls, set_to):
        cls.discount_use_reward = set_to

    @classmethod
    def set_simulator_instance(cls, set_to):
        cls.simulator_instance = set_to

    def configure_new(self, config):
        super(LanderGenome, self).configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super(LanderGenome, self).configure_crossover(genome1, genome2, config)
        if self.discount_use_reward:
            if genome1.reward is None:
                net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
                ignored_sc, reward, ignored_st = self.simulator_instance.simulate_genome(net=net1)
                genome1.reward = reward
            if genome2.reward is None:
                net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
                ignored_sc, reward, ignored_st = self.simulator_instance.simulate_genome(net=net2)
                genome2.reward = reward 
        if self.discount_use_reward and (genome1.reward
                                         is not None) and (genome2.reward
                                                           is not None):
            if genome1.reward > genome2.reward:
                self.discount = genome1.discount
            elif (genome1.reward < genome2.reward) or (random.random() < 0.5):
                self.discount = genome2.discount
            else:
                self.discount = genome1.discount
        elif random.random() < 0.5:
            self.discount = genome1.discount
        else:
            self.discount = genome2.discount


    def mutate(self, config):
        super(LanderGenome, self).mutate(config)
        self.discount += random.gauss(0.0, 0.025) # lowering SD - happens always
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super(LanderGenome, self).distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff

    def __str__(self):
        if self.reward is not None:
            return "Reward, discount: {0:n}; {1:n}\n{2!s}".format(self.reward,self.discount,
                                                  super(LanderGenome,self).__str__())
        else:
            return "Reward, discount: ?; {0:n}\n{1!s}".format(self.discount,
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

    return -1*np.sum(reward_error) / len(episodes)

class DoSimulation(object):
    def __init__(self, step_epsilon=True): # TODO: Make parallel via shared memory?
        self.curr_test_episodes = []
        self.num_simulations = 0
        self.total_simulation_length = 0
        self.step_epsilon = step_epsilon
        self.min_reward = -200 # will be updated
        self.max_reward = 200 # ditto

    def unload_test_episodes(self):
        to_return = self.curr_test_episodes[:] # better way to do this?
        self.curr_test_episodes = []
        return to_return

    def simulate_genome(self, net, use_seed=None):
        if use_seed is not None:
            env.seed(seed=use_seed)
        observation = env.reset()
        self.num_simulations += 1
        step = 0
        num_rand = 0
        data = []
        for_genome_reward = []
        while 1:
            step += 1
            do_rand = False
            if step < 200:
                if self.step_epsilon:
                    do_rand = bool(random.random() < 0.2)
                else:
                    do_rand = bool(random.random() < ((200-step)/500))
            if do_rand:
                action = env.action_space.sample()
            else:
                output = net.activate(observation)
                action = np.argmax(output)
                
            observation, reward, done, ignored_info = env.step(action)
            data.append(np.hstack((observation, action, reward)))
            if not do_rand:
                for_genome_reward.append(reward)
            else:
                num_rand += 1

            if done:
                break

        self.total_simulation_length += step

        reward = None
        if num_rand < step:
            reward = sum(for_genome_reward)*(step/(step-num_rand))
            reward = min(self.max_reward,max(self.min_reward,reward))
        data = np.array(data)
        score = np.sum(data[:,-1])
        self.curr_test_episodes.append((score, data))
        return (score, reward, step)

class PooledErrorCompute(object):
    def __init__(self, simulator, control_seed=False):
        if NUM_CORES < 2:
            self.pool = None
        else:
            self.pool = multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0
        self.control_seed = control_seed

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

        self.simulator = simulator
        simulator.min_reward = self.min_reward
        simulator.max_reward = self.max_reward

    def simulate(self, nets):
        scores = []
        if self.control_seed:
            use_seed = env.seed()
        else:
            use_seed = None
        for genome, net in nets:
            score, reward, step = self.simulator.simulate_genome(net=net,
                                                                 use_seed=use_seed)

            if reward is not None:
                genome.reward = reward
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)

            self.test_episodes += self.simulator.unload_test_episodes()

        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        nets = []
        for ignored_gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        print("network creation time {0:n}".format(time.time() - t0))
        t0 = time.time()

        last_num_test_episodes = len(self.test_episodes)
        self.test_episodes += self.simulator.unload_test_episodes()

        # Periodically generate a new set of episodes for comparison.
        if (self.generation % 10) == 1:
            num_episodes_keep = 300 + len(self.test_episodes) - last_num_test_episodes
            self.test_episodes = self.test_episodes[-1*num_episodes_keep:]
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
                genome.fitness = compute_fitness(genome,
                                                 net,
                                                 self.test_episodes,
                                                 self.min_reward,
                                                 self.max_reward)
        else:
            jobs = []
            for genome, net in nets:
                jobs.append(self.pool.apply_async(compute_fitness,
                    (genome, net,
                     self.test_episodes, self.min_reward, self.max_reward)))

            for job, (ignored_genome_id, genome) in zip(jobs, genomes):
                genome.fitness = job.get(timeout=None)

        print("final fitness compute time {0:n}\n".format(time.time() - t0))


def run(control_seed=False,filename_ext="svg",step_epsilon=True,discount_use_reward=False):
    """Main loop."""
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(LanderGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    simulator = DoSimulation(step_epsilon=step_epsilon)

    LanderGenome.set_discount_use_reward(discount_use_reward)
    LanderGenome.set_simulator_instance(simulator)
    

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute(simulator=simulator, control_seed=control_seed)
    while 1:
        try:
            # TODO: Why 5, not 10, since only does simulation every 10?
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

            mfs = sum(stats.get_fitness_tmean()[-5:]) / 5.0
            print("Average tmean(trim=0.25) fitness over last 5 generations: {0:n}".format(mfs))

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
                    # TODO: Option to use softmax on outputs and add up to determine action
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
                print("Solved; total length {0:n}, num simulations {1:n}".format(
                    simulator.total_simulation_length,
                    simulator.num_simulations))

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
