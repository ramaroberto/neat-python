"""Handles standard neural networks, either feedforward or recurrent."""
from neat.nn.feed_forward import FeedForwardNetwork
from neat.nn.recurrent import RecurrentNetwork

def create(genome, config):
    """Returns either a RecurrentNetwork or a FeedForwardNetwork for this genome based on maybe_recurrent."""
    if genome.maybe_recurrent:
        return RecurrentNetwork.create(genome, config)

    return FeedForwardNetwork.create(genome, config)



