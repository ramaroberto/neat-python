from __future__ import print_function, division

import os
import math
import random
import struct

from multiprocessing import Pool

try:
    from Pillow import Image, ImageChops, ImageFilter, ImageOps, ImageStat
except ImportError:
    from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
import numpy as np
import neat

from neat.math_util import mean, tmean
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
FULL_SCALE = 10

def evaluate_lowres(genome, config, scheme, do_transpose=True):
    if scheme == 'gray':
        image1 = eval_gray_image(genome, config, WIDTH, HEIGHT)
        image2 = eval_gray_image(genome, config,
                                 int(round((WIDTH/2.0),0)),
                                 int(round((HEIGHT/2.0),0)))
    elif scheme == 'color':
        image1 = eval_color_image(genome, config, WIDTH, HEIGHT)
        image2 = eval_color_image(genome, config,
                                 int(round((WIDTH/2.0),0)),
                                 int(round((HEIGHT/2.0),0)))
    else:
        raise ValueError('Unexpected scheme: {0!r}'.format(scheme))

    image1 = np.clip(np.array(image1), 0, 255).astype(np.uint8)
    image2 = np.clip(np.array(image2), 0, 255).astype(np.uint8)
    if do_transpose:
        if self.scheme == 'gray':
            image1 = Image.fromarray(image1, mode="L")
            image2 = Image.fromarray(image2, mode="L")
        else:
            image1 = Image.fromarray(image1, mode="RGB")
            image2 = Image.fromarray(image2, mode="RGB")
        image1_trans = {None:image1}
        image2_trans = {None:image2}
        for i in [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                  Image.ROTATE_90, Image.ROTATE_180,
                  Image.ROTATE_270, Image.TRANSPOSE]:
            image1_trans[i] = image1.transpose(i)
            image2_trans[i] = image2.transpose(i)
        return tuple([image1_trans,image2_trans])
    return tuple([image1,image2])

class NoveltyEvaluator(object):
    def __init__(self, num_workers, scheme):
        self.num_workers = num_workers
        self.scheme = scheme # note: 'mono' not usable
        self.pool = Pool(num_workers)
        self.archive = []
        self.out_index = 1
        if self.scheme == 'color':
            init_image = Image.new(mode="RGB",
                                   size=(WIDTH, HEIGHT),
                                   color=(127,127,127))
        else:
            init_image = Image.new(mode="L",
                                   size=(WIDTH,HEIGHT),
                                   color=127)
        self.add_to_archive(init_image, image_is_array=False)

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

    def flatten_image(self, image, is_array=False):
        if is_array:
            image_array = image
        else:
            image_array = self.array_from_image(image)
        float_image = image_array.astype(np.float32) / 255.0
        flattened = float_image.flatten()
        return flattened

    def get_kernel_edges(self, image, kernel, is_array=False,
                         want_flattened=True, autocontrast=False):
        if is_array:
            pillow_image = self.image_from_array(image)
        else:
            pillow_image = image
        pillow_image = pillow_image.filter(ImageFilter.UnsharpMask)
        stat = ImageStat.Stat(pillow_image)
        if self.scheme == 'color':
            R, G, B = stat.median
            pillow_image = ImageOps.expand(pillow_image, border=2, fill=(R,G,B))
        else:
            L = stat.median
            pillow_image = ImageOps.expand(pillow_image, border=2, fill=(L))
        pillow_image = pillow_image.filter(kernel)
        pillow_image = ImageOps.crop(pillow_image, border=2)
        if autocontrast:
            pillow_image = ImageOps.autocontrast(pillow_image)
        if not want_flattened:
            return pillow_image

        pillow_flattened = self.flatten_image(pillow_image)

        return (pillow_image, pillow_flattened)

    def get_base_edges(self, image, is_array=False, want_flattened=True):
        if is_array:
            pillow_image = self.image_from_array(image)
        else:
            pillow_image = image
        pillow_image = pillow_image.filter(ImageFilter.UnsharpMask)
        stat = ImageStat.Stat(pillow_image)
        if self.scheme == 'color':
            R, G, B = stat.median
            pillow_image = ImageOps.expand(pillow_image, border=1, fill=(R,G,B))
        else:
            L = stat.median
            pillow_image = ImageOps.expand(pillow_image, border=1, fill=(L))
        pillow_image = pillow_image.filter(ImageFilter.FIND_EDGES)
        pillow_image = ImageOps.crop(pillow_image, border=1)
        pillow_image = ImageOps.autocontrast(pillow_image)
        if not want_flattened:
            return pillow_image

        pillow_flattened = self.flatten_image(pillow_image)

        return (pillow_image, pillow_flattened)

    def get_base_edges_small(self, image2, is_array=False, want_flattened=True):
        if is_array:
            pillow2_image = self.image_from_array(image2)
        else:
            pillow2_image = image2
        pillow2_image = ImageOps.autocontrast(pillow2_image)
        stat = ImageStat.Stat(pillow2_image)
        if self.scheme == 'color':
            R, G, B = stat.median
            pillow2_image = ImageOps.expand(pillow2_image, border=1, fill=(R,G,B))
        else:
            L = stat.median
            pillow2_image = ImageOps.expand(pillow2_image, border=1, fill=(L))
        pillow2_image = pillow2_image.filter(ImageFilter.FIND_EDGES)
        pillow2_image = ImageOps.crop(pillow2_image, border=1)
        pillow2_image = ImageOps.autocontrast(pillow2_image)
        if not want_flattened:
            return pillow2_image

        pillow2_flattened = self.flatten_image(pillow2_image)

        return (pillow2_image, pillow2_flattened)

    def get_DoG(self, image, is_array=False, want_flattened=True, scale=2):
        if is_array:
            pillow_image = self.image_from_array(image)
        else:
            pillow_image = image

        scale2 = int(round((scale*1.6),0))
        assert scale < scale2, "Cannot do scale {0:n} (scale2 identical)".format(scale)

        width, height = pillow_image.size
        stat = ImageStat.Stat(pillow_image)
        if self.scheme == 'color':
            R, G, B = stat.median
            pillow_image = ImageOps.expand(pillow_image, border=scale2, fill=(R,G,B))
        else:
            L = stat.median
            pillow_image = ImageOps.expand(pillow_image, border=scale2, fill=(L))
        
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

    def add_to_archive(self, image_array, image2_array=None, image_is_array=False):
        if image_is_array:
            image = self.image_from_array(image_array)
        else:
            image = image_array
            image_array = self.array_from_image(image)
        if image2_array is None:
            image2 = image.resize((int(round((image.width/2.0),0)),
                                   int(round((image.height/2.0),0))),
                                  resample=Image.LANCZOS)
        else:
            image2 = self.image_from_array(image2_array)

        flattened = self.flatten_image(image_array, is_array=True)
        
        pillow_image, pillow_flattened = self.get_base_edges(image, is_array=False)
        pillow2_image, pillow2_flattened = self.get_base_edges_small(image2, is_array=False)
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
        self.archive.append((pillow_flattened,
                             pillow2_flattened,
                             pillowDoG2_flattened,
                             pillowDoG3_flattened,
                             pillowH_flattened,
                             pillowV_flattened,
                             pillowD1_flattened,
                             pillowD2_flattened,
                             flattened))
        return (pillow_image, pillow2_image,
                pillowDoG2_image, pillowDoG3_image,
                pillowH_image, pillowV_image,
                pillowD1_image, pillowD2_image)
        

    @staticmethod
    def count_DoG(values):
        return len([1 for x in values if x != (127.0/255.0)])

    def evaluate(self, genomes, config):
        jobs = []
        try:
            for ignored_genome_id, genome in genomes:
                jobs.append(self.pool.apply_async(evaluate_lowres, (genome, config, self.scheme)))
        except KeyboardInterrupt:
            self.pool.terminate()
            raise

        for (ignored_genome_id, genome), j in zip(genomes, jobs):
            try:
                image_trans, image2_trans = j.get()
            except KeyboardInterrupt:
                self.pool.terminate()
                raise

            genome.fitness = -1.0

            for trans_type in iterkeys(image_trans):
                image = image_trans[trans_type]
                image2 = image2_trans[trans_type]

                flattened = self.flatten_image(image)
                
                pillow_image, pillow_flattened = self.get_base_edges(image)
                ignored_pillow2_image, pillow2_flattened = self.get_base_edges_small(image2)
                pillowDoG2_image, pillowDoG2_flattened = self.get_DoG(image)
                pillowDoG3_image, pillowDoG3_flattened = self.get_DoG(image, scale=3)
                pillowH_image, pillowH_flattened = self.get_kernel_edges(image,
                                                                         kernel=SOBEL_H_KERNEL)
                pillowV_image, pillowV_flattened = self.get_kernel_edges(image,
                                                                         kernel=SOBEL_V_KERNEL)
                pillowD1_image, pillowD1_flattened = self.get_kernel_edges(image,
                                                                           kernel=SOBEL_D1_KERNEL)
                pillowD2_image, pillowD2_flattened = self.get_kernel_edges(image,
                                                                           kernel=SOBEL_D2_KERNEL)
                
                for a1, a2, aDoG2, aDoG3, aH, aV, aD1, aD2, a3 in self.archive:
                    pairs1 = zip(pillow_flattened, a1)
                    pairs2 = zip(pillow2_flattened, a2)
                    pairsDoG2 = zip(pillowDoG2_flattened, aDoG2)
                    pairsDoG3 = zip(pillowDoG3_flattened, aDoG3)
                    pairsH = zip(pillowH_flattened, aH)
                    pairsV = zip(pillowV_flattened, aV)
                    pairsD1 = zip(pillowD1_flattened, aD1)
                    pairsD2 = zip(pillowD2_flattened, aD2)
                    pairs3 = zip(flattened, a3)
                    dist1 = []
                    dist2 = []
                    distDoG2 = []
                    distDoG3 = []
                    distH = []
                    distV = []
                    distD1 = []
                    distD2 = []
                    dist0 = []
                    for i, j in pairs1:
                        dist1.append(float(abs(i-j)))
                    for i, j in pairs2:
                        dist2.append(float(abs(i-j)))
                    for i, j in pairsDoG2:
                        distDoG2.append(float(abs(i-j)))
                    for i, j in pairsDoG3:
                        distDoG3.append(float(abs(i-j)))
                    for i, j in pairsH:
                        distH.append(float(abs(i-j)))
                    for i, j in pairsV:
                        distV.append(float(abs(i-j)))
                    for i, j in pairsD1:
                        distD1.append(float(abs(i-j)))
                    for i, j in pairsD2:
                        distD2.append(float(abs(i-j)))
                    for i, j in pairs3:
                        dist0.append(float(abs(i-j))/(WIDTH*HEIGHT*2))
                    dist1_avg = math.fsum(dist1)/max(1,
                                                     np.count_nonzero(pillow_flattened),
                                                     np.count_nonzero(a1))
                    dist2_avg = math.fsum(dist2)/max(1,
                                                     np.count_nonzero(pillow2_flattened),
                                                     np.count_nonzero(a2))
                    distDoG2_avg = math.fsum(distDoG2)/max(1,
                                                           count_DoG(pillowDoG2_flattened),
                                                           count_DoG(aDoG2))
                    distDoG3_avg = math.fsum(distDoG3)/max(1,
                                                           count_DoG(pillowDoG3_flattened),
                                                           count_DoG(aDoG3))
                    distH_avg = math.fsum(distH)/max(1,
                                                     np.count_nonzero(pillowH_flattened),
                                                     np.count_nonzero(aH))
                    distV_avg = math.fsum(distV)/max(1,
                                                     np.count_nonzero(pillowV_flattened),
                                                     np.count_nonzero(aV))
                    distD1_avg = math.fsum(distD1)/max(1,
                                                       np.count_nonzero(pillowD1_flattened),
                                                       np.count_nonzero(aD1))
                    distD2_avg = math.fsum(distD2)/max(1,
                                                       np.count_nonzero(pillowD2_flattened),
                                                       np.count_nonzero(aD2))
                    adist = math.fsum(dist0 + [min(dist1_avg,dist2_avg), mean([dist1_avg,dist2_avg]),
                                               min(distH_avg,distV_avg,distD1_avg,distD2_avg),
                                               mean([distH_avg,distV_avg,distD1_avg,distD2_avg]),
                                               min(distDoG2_avg,distDoG3_avg),
                                               mean([distDoG2_avg,distDoG3_avg]),
                                               tmean([dist1_avg,dist2_avg,distH_avg,distV_avg,
                                                      distD1_avg,distD2_avg])])
                    if genome.fitness < 0.0:
                        genome.fitness = adist
                    else:
                        genome.fitness = min(genome.fitness, adist)

            chance = 0.25/len(genomes)
            if random.random() < chance:
                self.add_to_archive(image_trans[None],
                                    image2_array=self.array_from_image(image2_trans[None])


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

                im2 = self.get_base_edges(im, is_array=False, want_flattened=False)
                im2.save('novelty-{0:06d}-edge.png'.format(self.out_index))

                pillow_image.save('novelty-{0:06d}-edge-small.png'.format(self.out_index))
                pillowDoG2_image.save('novelty-{0:06d}-DoG2-small.png'.format(self.out_index))
                pillowDoG3_image.save('novelty-{0:06d}-DoG3-small.png'.format(self.out_index))
                pillowH_image.save('novelty-{0:06d}-edgeH-small.png'.format(self.out_index))
                pillowV_image.save('novelty-{0:06d}-edgeV-small.png'.format(self.out_index))
                pillowD1_image.save('novelty-{0:06d}-edgeD1-small.png'.format(self.out_index))
                pillowD2_image.save('novelty-{0:06d}-edgeD2-small.png'.format(self.out_index))

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

        im2 = ne.get_base_edges(im, is_array=False, want_flattened=False)
        im2.save('winning-novelty-{0:06d}-edge.png'.format(pop.generation))

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

        image, image2 = evaluate_lowres(winner, config, ne.scheme, do_transform=False)

        pillow_image, ignored, pillowDoG2_image, pillowDoG3_image, pillowH_image, pillowV_image, pillowD1_image, pillowD2_image = ne.add_to_archive(
            image, image2)

        pillow_image.save('winning-novelty-{0:06d}-edge-small.png'.format(pop.generation))
        pillowDoG2_image.save('winning-novelty-{0:06d}-DoG2-small.png'.format(pop.generation))
        pillowDoG3_image.save('winning-novelty-{0:06d}-DoG3-small.png'.format(pop.generation))
        pillowH_image.save('winning-novelty-{0:06d}-edgeH-small.png'.format(pop.generation))
        pillowV_image.save('winning-novelty-{0:06d}-edgeV-small.png'.format(pop.generation))
        pillowD1_image.save('winning-novelty-{0:06d}-edgeD1-small.png'.format(pop.generation))
        pillowD2_image.save('winning-novelty-{0:06d}-edgeD2-small.png'.format(pop.generation))


if __name__ == '__main__':
    run()
