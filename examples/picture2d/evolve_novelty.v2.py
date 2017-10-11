from __future__ import print_function, division

import cProfile
import os
import math
import random
import struct
import sys

from multiprocessing import Pool

try:
    from Pillow import Image, ImageChops, ImageFilter, ImageOps, ImageStat
except ImportError:
    from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
import numpy as np
import scipy.misc
import scipy.stats
import neat

from neat.math_util import mean, tmean, NORM_EPSILON
from neat.six_util import iterkeys
from common import eval_gray_image, eval_color_image

SOBEL_H_KERNEL = ImageFilter.Kernel(size=(5,5),
                                    scale=2,
                                    offset=127,
                                    kernel=[
                                        2.0, 1.0, 0.0, -1.0, -2.0,
                                        2.0, 1.0, 0.0, -1.0, -2.0,
                                        4.0, 2.0, 0.0, -2.0, -4.0,
                                        2.0, 1.0, 0.0, -1.0, -2.0,
                                        2.0, 1.0, 0.0, -1.0, -2.0
                                        ])

SOBEL_V_KERNEL = ImageFilter.Kernel(size=(5,5),
                                    scale=2,
                                    offset=127,
                                    kernel=[
                                        2.0,   2.0,  4.0,  2.0,  2.0,
                                        1.0,   1.0,  2.0,  1.0,  1.0,
                                        0.0,   0.0,  0.0,  0.0,  0.0,
                                        -1.0, -1.0, -2.0, -1.0, -1.0,
                                        -2.0, -2.0, -4.0, -2.0, -2.0
                                        ])

SOBEL_D1_KERNEL = ImageFilter.Kernel(size=(5,5),
                                     scale=2,
                                     offset=127,
                                     kernel=[
                                         4.0,  2.0,  2.0,  1.0,  0.0,
                                         2.0,  2.0,  1.0,  0.0, -1.0,
                                         2.0,  1.0,  0.0, -1.0, -2.0,
                                         1.0,  0.0, -1.0, -2.0, -2.0,
                                         0.0, -1.0, -2.0, -2.0, -4.0
                                         ])

SOBEL_D2_KERNEL = ImageFilter.Kernel(size=(5,5),
                                     scale=2,
                                     offset=127,
                                     kernel=[
                                          0.0,  1.0,  2.0,  2.0, 4.0,
                                         -1.0,  0.0,  1.0,  2.0, 2.0,
                                         -2.0, -1.0,  0.0,  1.0, 2.0,
                                         -2.0, -2.0, -1.0,  0.0, 1.0,
                                         -4.0, -2.0, -2.0, -1.0, 0.0
                                         ])

WIDTH, HEIGHT = 32, 32
FULL_SCALE = 16

def evaluate_lowres(genome, config, scheme):
    if scheme == 'gray':
        image1 = eval_gray_image(genome, config, WIDTH, HEIGHT)
    elif scheme == 'color':
        image1 = eval_color_image(genome, config, WIDTH, HEIGHT)
    else:
        raise ValueError('Unexpected scheme: {0!r}'.format(scheme))

    image1 = np.clip(np.array(image1), 0, 255).astype(np.uint8)

    return image1

class NoveltyEvaluator(object):
    def __init__(self, num_workers, scheme):
        self.num_workers = num_workers
        self.scheme = scheme # note: 'mono' not usable
        self.pool = Pool(num_workers)
        self.archive = []
        self.out_index = 1
        for num in [0, 127, 255]:
            if self.scheme == 'color':
                init_image = Image.new(mode="RGB",
                                       size=(WIDTH, HEIGHT),
                                       color=(num,num,num))
            else:
                init_image = Image.new(mode="L",
                                       size=(WIDTH,HEIGHT),
                                       color=num)
            self.add_to_archive(init_image, image_is_array=False, do_transpose=False)

    def image_from_array(self, image):
        if self.scheme == 'color':
            return Image.fromarray(image, mode="RGB")

        return Image.fromarray(image, mode="L")

    @staticmethod
    def array_from_image(image):
        return scipy.misc.fromimage(image)

##        image_bytes = image.tobytes()
##        array = []
##        for x in image_bytes:
##            array.append(struct.unpack("B",x))
##        return np.array(array).astype(np.uint8)

    def flatten_image(self, image, is_array=False):
        if is_array:
            image_array = image
        else:
            image_array = self.array_from_image(image)
        float_image = image_array.astype(np.float32) / 255.0
        return float_image.flatten()

    def add_border(self, orig_image, border=2):
        if border < 1:
            raise ValueError("Border must be at least 1, not {0!r}".format(border))
        stat = ImageStat.Stat(orig_image)
        if self.scheme == 'color':
            R, G, B = stat.median
            expanded_image = ImageOps.expand(orig_image, border=border, fill=(R,G,B))
        else:
            L = stat.median
            expanded_image = ImageOps.expand(orig_image, border=border, fill=(L))
        expanded_image = expanded_image.filter(ImageFilter.GaussianBlur(radius=(border+1)))
        expanded_image.paste(orig_image, box=(border,border))
        return expanded_image

    def get_kernel_edges(self, image, kernel, is_array=False,
                         want_flattened=True, autocontrast=False):
        if is_array:
            pillow_image = self.image_from_array(image)
        else:
            pillow_image = image
        pillow_image = pillow_image.filter(ImageFilter.UnsharpMask)
        pillow_image = self.add_border(pillow_image, border=2)
        pillow_image = pillow_image.filter(kernel)
        pillow_image = ImageOps.crop(pillow_image, border=2)
        if autocontrast:
            pillow_image = ImageOps.autocontrast(pillow_image)
        if not want_flattened:
            return pillow_image

        pillow_flattened = self.flatten_image(pillow_image)

        return (pillow_image, pillow_flattened)

##    def get_base_edges(self, image, is_array=False, want_flattened=True):
##        if is_array:
##            pillow_image = self.image_from_array(image)
##        else:
##            pillow_image = image
##        pillow_image = pillow_image.filter(ImageFilter.UnsharpMask)
##        stat = ImageStat.Stat(pillow_image)
##        if self.scheme == 'color':
##            R, G, B = stat.median
##            pillow_image = ImageOps.expand(pillow_image, border=1, fill=(R,G,B))
##        else:
##            L = stat.median
##            pillow_image = ImageOps.expand(pillow_image, border=1, fill=(L))
##        pillow_image = pillow_image.filter(ImageFilter.FIND_EDGES)
##        pillow_image = ImageOps.crop(pillow_image, border=1)
##        pillow_image = ImageOps.autocontrast(pillow_image)
##        if not want_flattened:
##            return pillow_image

##        pillow_flattened = self.flatten_image(pillow_image)

##        return (pillow_image, pillow_flattened)

##    def get_base_edges_small(self, image2, is_array=False, want_flattened=True):
##        if is_array:
##            pillow2_image = self.image_from_array(image2)
##        else:
##            pillow2_image = image2
##        pillow2_image = ImageOps.autocontrast(pillow2_image)
##        stat = ImageStat.Stat(pillow2_image)
##        if self.scheme == 'color':
##            R, G, B = stat.median
##            pillow2_image = ImageOps.expand(pillow2_image, border=1, fill=(R,G,B))
##        else:
##            L = stat.median
##            pillow2_image = ImageOps.expand(pillow2_image, border=1, fill=(L))
##        pillow2_image = pillow2_image.filter(ImageFilter.FIND_EDGES)
##        pillow2_image = ImageOps.crop(pillow2_image, border=1)
##        pillow2_image = ImageOps.autocontrast(pillow2_image)
##        if not want_flattened:
##            return pillow2_image

##        pillow2_flattened = self.flatten_image(pillow2_image)

##        return (pillow2_image, pillow2_flattened)

    def get_DoG(self, image, is_array=False, want_flattened=True, scale=2):
        if is_array:
            pillow_image = self.image_from_array(image)
        else:
            pillow_image = image

        scale2 = int(round((scale*1.6),0))
        assert scale < scale2, "Cannot do scale {0:n} (scale2 identical)".format(scale)

        #width, height = pillow_image.size
        pillow_image = self.add_border(pillow_image, border=scale2)
        
        pillow1_image = pillow_image.filter(ImageFilter.GaussianBlur(scale))
        pillow2_image = pillow_image.filter(ImageFilter.GaussianBlur(scale2))
        pillow_image = ImageChops.subtract(pillow1_image,pillow2_image,scale=2,offset=127)

        section_size = scale2
        if (section_size % 2) == 0:
            section_size += 1

        max_pillow_image = pillow_image.filter(ImageFilter.MaxFilter(section_size))
        min_pillow_image = pillow_image.filter(ImageFilter.MinFilter(section_size))

        pillow_image = ImageOps.crop(pillow_image, border=scale2)
        max_pillow_image = ImageOps.crop(max_pillow_image, border=scale2)
        min_pillow_image = ImageOps.crop(min_pillow_image, border=scale2)

        stat2 = ImageStat.Stat(pillow_image)
        if self.scheme == 'color':
            R_med, G_med, B_med = stat2.median
            R_mean, G_mean, B_mean = stat2.mean
            (R_upper, ignored), (G_upper, ignored), (B_upper, ignored) = max_pillow_image.getextrema()
            R_upper = max(R_upper,127,int(math.ceil(max(R_med,R_mean))))
            G_upper = max(G_upper,127,int(math.ceil(max(G_med,G_mean))))
            B_upper = max(B_upper,127,int(math.ceil(max(B_med,B_mean))))
            (ignored, R_lower), (ignored, G_lower), (ignored, B_lower) = min_pillow_image.getextrema()
            R_lower = min(R_lower,127,int(math.floor(min(R_med,R_mean))))
            G_lower = min(G_lower,127,int(math.floor(min(G_med,G_mean))))
            B_lower = min(B_lower,127,int(math.floor(min(B_med,B_mean))))
            R_image, G_image, B_image = pillow_image.split()
            R_lut = []
            for i in range(256):
                if i >= R_upper:
                    R_lut.append(i)
                elif i <= R_lower:
                    R_lut.append(i)
                else:
                    R_lut.append(127)
            R_image = R_image.point(R_lut)
            G_lut = []
            for i in range(256):
                if i >= G_upper:
                    G_lut.append(i)
                elif i <= G_lower:
                    G_lut.append(i)
                else:
                    G_lut.append(127)
            G_image = G_image.point(G_lut)
            B_lut = []
            for i in range(256):
                if i >= B_upper:
                    B_lut.append(i)
                elif i <= B_lower:
                    B_lut.append(i)
                else:
                    B_lut.append(127)
            B_image = B_image.point(B_lut)
            pillow_image = Image.merge(mode="RGB",bands=[R_image,G_image,B_image])
        else:
            L_med = stat2.median
            L_mean = stat2.mean
            L_upper, ignored = max_pillow_image.getextrema()
            L_upper = max(L_upper,127,int(math.ceil(max(L_med,L_mean))))
            ignored, L_lower = min_pillow_image.getextrema()
            L_lower = min(L_lower,127,int(math.floor(min(L_med,L_mean))))
            L_lut = []
            for i in range(256):
                if i >= L_upper:
                    L_lut.append(i)
                elif i <= L_lower:
                    L_lut.append(i)
                else:
                    L_lut.append(127)
            pillow_image = pillow_image.point(L_lut)

        if not want_flattened:
            return pillow_image

        pillow_flattened = self.flatten_image(pillow_image)

        return (pillow_image, pillow_flattened)

    def _add_to_archive_inner(self, image):
        flattened = self.flatten_image(image, is_array=False)
        
        #pillow_image, pillow_flattened = self.get_base_edges(image, is_array=False)
        #pillow2_image, pillow2_flattened = self.get_base_edges_small(image2, is_array=False)
        pillowDoG2_image, pillowDoG2_flattened = self.get_DoG(image, is_array=False)
        pillowDoG3_image, pillowDoG3_flattened = self.get_DoG(image, is_array=False, scale=3)
        pillowH_image, pillowH_flattened = self.get_kernel_edges(image,
                                                                 kernel=SOBEL_H_KERNEL,
                                                                 is_array=False)
        pillowV_image, pillowV_flattened = self.get_kernel_edges(image,
                                                                 kernel=SOBEL_V_KERNEL,
                                                                 is_array=False)
        pillowD1_image, pillowD1_flattened = self.get_kernel_edges(image,
                                                                   kernel=SOBEL_D1_KERNEL,
                                                                   is_array=False)
        pillowD2_image, pillowD2_flattened = self.get_kernel_edges(image,
                                                                   kernel=SOBEL_D2_KERNEL,
                                                                   is_array=False)
        self.archive.append((pillowDoG2_flattened,
                             pillowDoG3_flattened,
                             pillowH_flattened,
                             pillowV_flattened,
                             pillowD1_flattened,
                             pillowD2_flattened,
                             flattened))
        return (pillowDoG2_image, pillowDoG3_image,
                pillowH_image, pillowV_image,
                pillowD1_image, pillowD2_image)

    def add_to_archive(self, image_array, image_is_array=False,
                       do_transpose=True):
        if image_is_array:
            image = self.image_from_array(image_array)
        else:
            image = image_array

        if do_transpose:
            for i in [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                      Image.ROTATE_90, Image.ROTATE_180,
                      Image.ROTATE_270, Image.TRANSPOSE]:
                self._add_to_archive_inner(image.transpose(i))
        return self._add_to_archive_inner(image)


    def evaluate(self, genomes, config):
        jobs = []
        try:
            for ignored_genome_id, genome in genomes:
                jobs.append(self.pool.apply_async(evaluate_lowres, (genome, config, self.scheme)))
        except KeyboardInterrupt:
            self.pool.terminate()
            raise

        for_archive = []
        include_chance = 1/len(genomes)

        for (ignored_genome_id, genome), j in zip(genomes, jobs):
            try:
                image_array = j.get()
            except KeyboardInterrupt:
                self.pool.terminate()
                raise

            genome.fitness = -1.0
            genome_dists = []

            flattened = self.flatten_image(image_array, is_array=True)
            image = self.image_from_array(image_array)
            
            #pillow_image, pillow_flattened = self.get_base_edges(image)
            #ignored_pillow2_image, pillow2_flattened = self.get_base_edges_small(image2_array, is_array=True)
            ignored, pillowDoG2_flattened = self.get_DoG(image)
            DoG2_nonzero = np.count_nonzero(pillowDoG2_flattened != (127.0/255.0))
            ignored, pillowDoG3_flattened = self.get_DoG(image, scale=3)
            DoG3_nonzero = np.count_nonzero(pillowDoG3_flattened != (127.0/255.0))
            ignored, pillowH_flattened = self.get_kernel_edges(image,
                                                               kernel=SOBEL_H_KERNEL)
            H_nonzero = np.count_nonzero(pillowH_flattened != (127.0/255.0))
            ignored, pillowV_flattened = self.get_kernel_edges(image,
                                                               kernel=SOBEL_V_KERNEL)
            V_nonzero = np.count_nonzero(pillowV_flattened != (127.0/255.0))
            ignored, pillowD1_flattened = self.get_kernel_edges(image,
                                                                kernel=SOBEL_D1_KERNEL)
            D1_nonzero = np.count_nonzero(pillowD1_flattened != (127.0/255.0))
            ignored, pillowD2_flattened = self.get_kernel_edges(image,
                                                                kernel=SOBEL_D2_KERNEL)
            D2_nonzero = np.count_nonzero(pillowD2_flattened != (127.0/255.0))
            for aDoG2, aDoG3, aH, aV, aD1, aD2, a3 in self.archive: # TODO: save count_nonzero results
                dist0 = np.abs(flattened - a3)
                curr_dist = np.sum(dist0)/(WIDTH*HEIGHT)
                if curr_dist < NORM_EPSILON:
                    genome.fitness = 0.0
                    break
                distH = np.abs(pillowH_flattened - aH)
                distV = np.abs(pillowV_flattened - aV)
                distD1 = np.abs(pillowD1_flattened - aD1)
                distD2 = np.abs(pillowD2_flattened - aD2)

                distH_avg = np.sum(distH)/max(1,
                                              H_nonzero,
                                              np.count_nonzero(aH != (127.0/255.0)))
                distV_avg = np.sum(distV)/max(1,
                                              V_nonzero,
                                              np.count_nonzero(aV != (127.0/255.0)))
                distD1_avg = np.sum(distD1)/max(1,
                                                D1_nonzero,
                                                np.count_nonzero(aD1 != (127.0/255.0)))
                distD2_avg = np.sum(distD2)/max(1,
                                                D2_nonzero,
                                                np.count_nonzero(aD2 != (127.0/255.0)))
                edge_dist = max(distH_avg, distV_avg, distD1_avg, distD2_avg)
                curr_dist *= edge_dist
                if curr_dist < NORM_EPSILON:
                    genome.fitness = 0.0
                    break
                distDoG2 = np.abs(pillowDoG2_flattened - aDoG2)
                distDoG3 = np.abs(pillowDoG3_flattened - aDoG3)
                distDoG2_avg = np.sum(distDoG2)/max(1,
                                                    DoG2_nonzero,
                                                    np.count_nonzero(aDoG2 != (127.0/255.0)))
                distDoG3_avg = np.sum(distDoG3)/max(1,
                                                    DoG3_nonzero,
                                                    np.count_nonzero(aDoG3 != (127.0/255.0)))
                curr_dist *= max(distDoG2_avg, distDoG3_avg)
                if curr_dist < sys.float_info.epsilon:
                    genome.fitness = 0.0
                    break
                else:
                    genome_dists.append(curr_dist)

            if genome.fitness == 0.0:
                continue

            genome_dists.sort()
            # 7 is due to transforms
            num_want = int(min((15*7),math.ceil(len(genome_dists)*0.5))) # long-term, want to check vs current population also
            genome.fitness = scipy.stats.gmean(genome_dists[:num_want])

            if (random.random() < include_chance) and (genome.fitness > NORM_EPSILON):
                for_archive.append(tuple([image,genome]))

        for image, genome in for_archive:
            self.add_to_archive(image)

            if self.scheme == 'gray':
                image = eval_gray_image(genome, config, FULL_SCALE * WIDTH,
                                        FULL_SCALE * HEIGHT)
            elif self.scheme == 'color':
                image = eval_color_image(genome, config, FULL_SCALE * WIDTH,
                                         FULL_SCALE * HEIGHT)
            else:
                raise ValueError('Unexpected scheme: {0!r}'.format(self.scheme))

            im_array = np.clip(np.array(image), 0, 255).astype(np.uint8)
            im = self.image_from_array(im_array)
            im_exp = ImageOps.autocontrast(im)
            im_exp.save('novelty-{0:06d}.png'.format(self.out_index))

            self.out_index += 1

        print('{0:n} archive entries'.format(len(self.archive)))


def run():
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'novelty_config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    ne = NoveltyEvaluator(4, 'color') # do not use 'mono'
    if ne.scheme == 'color':
        config.output_nodes = 3
    else:
        config.output_nodes = 1

    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))

    while 1:
        if len(ne.archive) >= (7*15):
            cProfile.runctx("pop.run(ne.evaluate, 1)", globals=globals(), locals=locals(), sort="cumtime")
            break
        winner = pop.run(ne.evaluate, 1)
        print('\nBest genome:\n{!s}'.format(winner))
        pop.best_genome = None # since fitnesses can change!

        if ne.scheme == 'gray':
            image = eval_gray_image(winner, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
        elif ne.scheme == 'color':
            image = eval_color_image(winner, config, FULL_SCALE * WIDTH, FULL_SCALE * HEIGHT)
        else:
            raise ValueError('Unexpected scheme: {0!r}'.format(ne.scheme))

        im_array = np.clip(np.array(image), 0, 255).astype(np.uint8)
        im = ne.image_from_array(im_array)
        im_exp = ImageOps.autocontrast(im)
        im_exp.save('winning-novelty-{0:06d}.png'.format(pop.generation))

##        im2 = ne.get_base_edges(im, is_array=False, want_flattened=False)
##        im2_exp = ImageOps.autocontrast(im2)
##        im2_exp.save('winning-novelty-{0:06d}-edge.png'.format(pop.generation))

        imDoG = ne.get_DoG(im, is_array=False, want_flattened=False, scale=(2*FULL_SCALE))
        imDoG = ImageOps.autocontrast(imDoG)
        imDoG.save('winning-novelty-{0:06d}-DoG.png'.format(pop.generation))

        imH = ne.get_kernel_edges(im, kernel=SOBEL_H_KERNEL, is_array=False,
                                  want_flattened=False, autocontrast=True)
        imH.save('winning-novelty-{0:06d}-edgeH.png'.format(pop.generation))
        imV = ne.get_kernel_edges(im, kernel=SOBEL_V_KERNEL, is_array=False,
                                  want_flattened=False, autocontrast=True)
        imV.save('winning-novelty-{0:06d}-edgeV.png'.format(pop.generation))
        imD1 = ne.get_kernel_edges(im, kernel=SOBEL_D1_KERNEL, is_array=False,
                                   want_flattened=False, autocontrast=True)
        imD1.save('winning-novelty-{0:06d}-edgeD1.png'.format(pop.generation))
        imD2 = ne.get_kernel_edges(im, kernel=SOBEL_D2_KERNEL, is_array=False,
                                   want_flattened=False, autocontrast=True)
        imD2.save('winning-novelty-{0:06d}-edgeD2.png'.format(pop.generation))

        image_array = evaluate_lowres(winner, config, ne.scheme)

        pillowDoG2_image, pillowDoG3_image, pillowH_image, pillowV_image, pillowD1_image, pillowD2_image = ne.add_to_archive(
            image_array, image_is_array=True)

        pillowDoG2_image.save('winning-novelty-{0:06d}-DoG2-small.png'.format(pop.generation))
        pillowDoG3_image.save('winning-novelty-{0:06d}-DoG3-small.png'.format(pop.generation))
        pillowH_image.save('winning-novelty-{0:06d}-edgeH-small.png'.format(pop.generation))
        pillowV_image.save('winning-novelty-{0:06d}-edgeV-small.png'.format(pop.generation))
        pillowD1_image.save('winning-novelty-{0:06d}-edgeD1-small.png'.format(pop.generation))
        pillowD2_image.save('winning-novelty-{0:06d}-edgeD2-small.png'.format(pop.generation))


if __name__ == '__main__':
    run()
