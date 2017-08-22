import glob
import os

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import neat
from neat.six_util import iterkeys

def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration*')
    filenames_to_change_first = {}
    filenames_to_change_second = {}
    for filename in glob.glob(config_path):
        if "_tmp" in filename:
            continue
        if "_use_defaults" in filename:
            continue
        try:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 filename)
        except configparser.NoSectionError:
            continue
        filenames_to_change_first[filename] = filename + "_use_defaults"
        tmp_filename = filename + "_tmp"
        filenames_to_change_second[tmp_filename] = filename
        config.save(tmp_filename)

    for filename in iterkeys(filenames_to_change_first):
        os.rename(filename, filenames_to_change_first[filename])
    for filename in iterkeys(filenames_to_change_second):
        os.rename(filename, filenames_to_change_second[filename])


if __name__ == '__main__':
    main()
