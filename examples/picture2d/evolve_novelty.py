from __future__ import print_function, division


import os
import math
import random

from multiprocessing import Pool

try:
    from Pillow import Image
except ImportError:
    from PIL import Image
import numpy as np
import neat

from common import eval_mono_image, eval_gray_image, eval_color_image

WIDTH, HEIGHT = 32, 32
FULL_SCALE = 16

def evaluate_lowres(genome, config, scheme):
    if scheme == 'gray':
        return eval_gray_image(genome, config, WIDTH, HEIGHT)
    elif scheme == 'color':
        return eval_color_image(genome, config, WIDTH, HEIGHT)
    elif scheme == 'mono':
        return eval_mono_image(genome, config, WIDTH, HEIGHT)

    raise Exception('Unexpected scheme: {0!r}'.format(scheme))


class NoveltyEvaluator(object):
    def __init__(self, num_workers, scheme):
        self.num_workers = num_workers
        self.scheme = scheme
        self.pool = Pool(num_workers)
        self.archive = []
        self.out_index = 1

    def image_from_array(self, image):
        if self.scheme == 'color':
            return Image.fromarray(image, mode="RGB")

        return Image.fromarray(image, mode="L")

    def evaluate(self, genomes, config):
        jobs = []
        try:
            for genome_id, genome in genomes:
                jobs.append(self.pool.apply_async(evaluate_lowres, (genome, config, self.scheme)))
        except KeyboardInterrupt:
            self.pool.terminate()
            raise

        new_archive_entries = []
        for (genome_id, genome), j in zip(genomes, jobs):
            try:
                image = np.clip(np.array(j.get()), 0, 255).astype(np.uint8)
            except KeyboardInterrupt:
                self.pool.terminate()
                raise

            float_image = image.astype(np.float32) / 255.0

            genome.fitness = -1.0
            num_archive = 0
            flattened = float_image.flatten()
            for a in self.archive:
                pairs = zip(flattened, a)
                dist = []
                for i, j in pairs:
                    dist.append(float(abs(i-j)))
                adist = math.fsum(dist)
                #adist = float(np.linalg.norm(float_image.ravel() - a.ravel(), ord=2))
                if genome.fitness < 0.0:
                    genome.fitness = adist
                else:
                    genome.fitness = min(genome.fitness, adist)
                num_archive += 1
            print("Processed vs {0:n} in archive (fitness {1:n})".format(
                num_archive,genome.fitness))

            chance = 1.0/len(genomes)
            if random.random() < chance:
                new_archive_entries.append(flattened)
                #im = self.image_from_array(image)
                #im.save("novelty-{0:06d}.png".format(self.out_index))

                if self.scheme == 'gray':
                    image = eval_gray_image(genome, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
                elif self.scheme == 'color':
                    image = eval_color_image(genome, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
                elif self.scheme == 'mono':
                    image = eval_mono_image(genome, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
                else:
                    raise Exception('Unexpected scheme: {0!r}'.format(self.scheme))

                im = np.clip(np.array(image), 0, 255).astype(np.uint8)
                im = self.image_from_array(im)
                im.save('novelty-{0:06d}.png'.format(self.out_index))

                self.out_index += 1

        self.archive.extend(new_archive_entries)
        print('{0} archive entries'.format(len(self.archive)))


def run():
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'novelty_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    ne = NoveltyEvaluator(4, 'color')
    if ne.scheme == 'color':
        config.output_nodes = 3
    else:
        config.output_nodes = 1

    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))

    while 1:
        winner = pop.run(ne.evaluate, 1)
        pop.best_genome = None # since fitnesses can change!

        if ne.scheme == 'gray':
            image = eval_gray_image(winner, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
        elif ne.scheme == 'color':
            image = eval_color_image(winner, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
        elif ne.scheme == 'mono':
            image = eval_mono_image(winner, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
        else:
            raise Exception('Unexpected scheme: {0!r}'.format(ne.scheme))

        im = np.clip(np.array(image), 0, 255).astype(np.uint8)
        im = ne.image_from_array(im)
        im.save('winning-novelty-{0:06d}.png'.format(pop.generation))

        if ne.scheme == 'gray':
            image = eval_gray_image(winner, config, WIDTH, HEIGHT)
        elif ne.scheme == 'color':
            image = eval_color_image(winner, config, WIDTH, HEIGHT)
        elif ne.scheme == 'mono':
            image = eval_mono_image(winner, config, WIDTH, HEIGHT)
        else:
            raise Exception('Unexpected scheme: {0!r}'.format(ne.scheme))

        im = np.clip(np.array(image), 0, 255).astype(np.uint8)
        float_image = im.astype(np.float32) / 255.0
        ne.archive.append(float_image.flatten())


if __name__ == '__main__':
    run()
