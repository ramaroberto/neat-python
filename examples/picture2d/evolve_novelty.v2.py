from __future__ import print_function, division

import os
import math
import random
import struct

from multiprocessing import Pool

try:
    from Pillow import Image, ImageFilter, ImageOps
except ImportError:
    from PIL import Image, ImageFilter, ImageOps
import numpy as np
import neat

from common import eval_mono_image, eval_gray_image, eval_color_image

WIDTH, HEIGHT = 32, 32
FULL_SCALE = 8

def evaluate_lowres(genome, config, scheme):
    if scheme == 'gray':
        return tuple([eval_gray_image(genome, config, WIDTH, HEIGHT),
                      eval_gray_image(genome, config, int(WIDTH/2), int(HEIGHT/2))])
    elif scheme == 'color':
        return tuple([eval_color_image(genome, config, WIDTH, HEIGHT),
                      eval_color_image(genome, config, int(WIDTH/2), int(HEIGHT/2))])
    elif scheme == 'mono':
        return tuple([eval_mono_image(genome, config, WIDTH, HEIGHT),
                      eval_mono_image(genome, config, int(WIDTH/2), int(HEIGHT/2))])

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

    @staticmethod
    def array_from_image(image):
        image_bytes = image.tobytes()
        array = []
        for x in image_bytes:
            array.append(struct.unpack("B",x))
        return np.array(array).astype(np.uint8)

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
                init_tuple = j.get()
            except KeyboardInterrupt:
                self.pool.terminate()
                raise

            image = np.clip(np.array(init_tuple[0]), 0, 255).astype(np.uint8)
            image2 = np.clip(np.array(init_tuple[1]), 0, 255).astype(np.uint8)
            float_image = image.astype(np.float32) / 255.0

            pillow_image = self.image_from_array(image)
            pillow_image = pillow_image.filter(ImageFilter.UnsharpMask)
            pillow_image = pillow_image.filter(ImageFilter.FIND_EDGES)
            if self.scheme != 'mono':
                pillow_image = ImageOps.autocontrast(pillow_image)
            pillow_array = self.array_from_image(pillow_image)
            float_pillow_image = pillow_array.astype(np.float32) / 255.0

            pillow2_image = self.image_from_array(image2)
            if self.scheme != 'mono':
                pillow2_image = ImageOps.autocontrast(pillow2_image)
            pillow2_image = pillow2_image.filter(ImageFilter.FIND_EDGES)
            if self.scheme != 'mono':
                pillow2_image = ImageOps.autocontrast(pillow2_image)
            pillow2_array = self.array_from_image(pillow2_image)
            float_pillow2_image = pillow2_array.astype(np.float32) / 255.0

            genome.fitness = -1.0
            pillow_flattened = float_pillow_image.flatten()
            pillow2_flattened = float_pillow2_image.flatten()
            flattened = float_image.flatten()
            for a1, a2, a3 in self.archive:
                pairs1 = zip(pillow_flattened, a1)
                pairs2 = zip(pillow2_flattened, a2)
                pairs3 = zip(flattened, a3)
                dist = []
                for i, j in pairs1:
                    dist.append(float(abs(i-j)))
                for i, j in pairs2:
                    dist.append(float(abs(i-j))*4)
                for i, j in pairs3:
                    dist.append(float(abs(i-j))/(WIDTH*HEIGHT)) # max is actually 2* this
                adist = math.fsum(dist)
                #adist = float(np.linalg.norm(float_image.ravel() - a.ravel(), ord=2))
                if genome.fitness < 0.0:
                    genome.fitness = adist
                else:
                    genome.fitness = min(genome.fitness, adist)

            chance = 1.0/len(genomes)
            if random.random() < chance:
                new_archive_entries.append((pillow_flattened,pillow2_flattened,pillow_flattened))
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
                im = im.filter(ImageFilter.UnsharpMask)
                im = im.filter(ImageFilter.FIND_EDGES)
                if self.scheme != 'mono':
                    im = ImageOps.autocontrast(im)
                im.save('novelty-{0:06d}-edge.png'.format(self.out_index))

                pillow_image.save('novelty-{0:06d}-edge-small.png'.format(self.out_index))

                self.out_index += 1

        self.archive.extend(new_archive_entries)
        print('{0:n} archive entries'.format(len(self.archive)))


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

        im = im.filter(ImageFilter.UnsharpMask)
        im = im.filter(ImageFilter.FIND_EDGES)
        im = ImageOps.autocontrast(im)
        im.save('winning-novelty-{0:06d}-edge.png'.format(pop.generation))

        if ne.scheme == 'gray':
            image = eval_gray_image(winner, config, WIDTH, HEIGHT)
            image2 = eval_gray_image(winner, config, int(WIDTH/2), int(HEIGHT/2))
        elif ne.scheme == 'color':
            image = eval_color_image(winner, config, WIDTH, HEIGHT)
            image2 = eval_color_image(winner, config, int(WIDTH/2), int(HEIGHT/2))
        elif ne.scheme == 'mono':
            image = eval_mono_image(winner, config, WIDTH, HEIGHT)
            image2 = eval_mono_image(winner, config, int(WIDTH/2), int(HEIGHT/2))
        else:
            raise Exception('Unexpected scheme: {0!r}'.format(ne.scheme))

        im = np.clip(np.array(image), 0, 255).astype(np.uint8)
        float_image = im.astype(np.float32) / 255.0

        pillow_image = ne.image_from_array(im)
        pillow_image = pillow_image.filter(ImageFilter.UnsharpMask)
        pillow_image = pillow_image.filter(ImageFilter.FIND_EDGES)
        if ne.scheme != 'mono':
            pillow_image = ImageOps.autocontrast(pillow_image)
        pillow_image.save('winning-novelty-{0:06d}-edge-small.png'.format(pop.generation))
        pillow_array = ne.array_from_image(pillow_image)
        float_pillow_image = pillow_array.astype(np.float32) / 255.0

        im = np.clip(np.array(image2), 0, 255).astype(np.uint8)
        pillow2_image = ne.image_from_array(im)
        if ne.scheme != 'mono':
            pillow2_image = ImageOps.autocontrast(pillow2_image)
        pillow2_image = pillow2_image.filter(ImageFilter.FIND_EDGES)
        if ne.scheme != 'mono':
            pillow2_image = ImageOps.autocontrast(pillow2_image)
        pillow2_array = ne.array_from_image(pillow2_image)
        float_pillow2_image = pillow2_array.astype(np.float32) / 255.0

        ne.archive.append((float_pillow_image.flatten(),float_pillow2_image.flatten(),float_image.flatten()))


if __name__ == '__main__':
    run()
