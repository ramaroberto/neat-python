"""Handles genes coding for node and connection attributes."""
import warnings
from random import random

from neat.attributes import FloatAttribute, BoolAttribute, FuncAttribute

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from neat.config import ConfigParameter # pylint: disable=unused-import
    from neat.genome import DefaultGenomeConfig # pylint: disable=unused-import

# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """
    def __init__(self, key): # type: (GeneKey) -> None
        self.key = key

    def __str__(self): # type: () -> str
        attrib = ['key'] + [a.name for a in self._gene_attributes] # type: ignore
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))


    def __lt__(self, other): # type: (BaseGene) -> bool
        assert isinstance(self.key,type(other.key)), "Cannot compare keys {0!r} and {1!r}".format(self.key,other.key)
        return self.key < other.key # type: ignore

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls): # type: () -> List[ConfigParameter]
        params = [] # type: List[ConfigParameter]
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__,cls),
                DeprecationWarning)
        for a in cls._gene_attributes: # type: ignore
            params += a.get_config_params()
        return params

    def init_attributes(self, config): # type: (DefaultGenomeConfig) -> None
        for a in self._gene_attributes: # type: ignore
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config): # type: (DefaultGenomeConfig) -> None
        for a in self._gene_attributes: # type: ignore
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self): # type: () -> Union[BaseGene, DefaultConnectionGene, DefaultNodeGene] # XXX
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes: # type: ignore
            if hasattr(a, 'copy'):
                setattr(new_gene, a.name, a.copy())
            else:
                setattr(new_gene, a.name, getattr(self, a.name))
        return new_gene

    def crossover(self,
                  gene2 # type: Union[BaseGene, DefaultConnectionGene, DefaultNodeGene] # XXX
                  ):
        # type: (...) -> Union[BaseGene, DefaultConnectionGene, DefaultNodeGene] # XXX
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes: # type: ignore
            if random() > 0.5:
                if hasattr(a, 'copy'):
                    setattr(new_gene, a.name, a.copy())
                else:
                    setattr(new_gene, a.name, getattr(self, a.name))
            else:
                gene2_attr = getattr(gene2, a.name)
                if hasattr(gene2_attr, 'copy'):
                    setattr(new_gene, a.name, gene2_attr.copy())
                else:
                    setattr(new_gene, a.name, gene2_attr)
        return new_gene


# TODO: Should these be in the nn module? iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        FuncAttribute('activation', options='sigmoid'),
                        FuncAttribute('aggregation', options='sum')]

    def __init__(self, key): # type: (NodeKey) -> None
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config): # type: (DefaultNodeGene, DefaultGenomeConfig) -> float
        """Returns the genetic distance between two node genes."""
        d = abs(self.bias - other.bias) + abs(self.response - other.response) # type: ignore
        if hasattr(self.activation, 'distance'): # type: ignore
            d += self.activation.distance(other.activation) # type: ignore
        elif self.activation != other.activation: # type: ignore
            d += 1.0

        if hasattr(self.aggregation, 'distance'): # type: ignore
            d += self.aggregation.distance(other.aggregation) # type: ignore
        elif self.aggregation != other.aggregation: # type: ignore
            d += 1.0

        return d * config.compatibility_weight_coefficient # type: ignore


# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def __init__(self, key): # type: (ConnKey) -> None
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self,
                 other, # type: DefaultConnectionGene
                 config # type: DefaultGenomeConfig
                 ):
        # type: (...) -> float
        """Returns the genetic distance between two connection genes."""
        d = abs(self.weight - other.weight) # type: ignore
        if self.enabled != other.enabled: # type: ignore
            d += 1.0
        return d * config.compatibility_weight_coefficient # type: ignore
