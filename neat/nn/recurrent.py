from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from neat.mypy_util import DefaultGenome, Config, DefaultGenomeConfig # pylint: disable=unused-import
    from neat.multiparameter import NormActFunc, NormAgFunc # pylint: disable=unused-import

class RecurrentNetwork(object):
    def __init__(self,
                 inputs, # type: List[NodeKey]
                 outputs, # type: List[NodeKey]
                 node_evals # type: List[Tuple[NodeKey, NormActFunc, NormAgFunc, float, float, List[Tuple[NodeKey, float]]]]
                 ):
        # type: (...) -> None
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}] # type: List[Dict[NodeKey, float]]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0

        self.active = 0 # type: int # c_type: c_uint

    def reset(self): # type: () -> None
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs): # type: (List[float]) -> List[float]
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active] # type: Dict[NodeKey, float]
        ovalues = self.values[1 - self.active] # type: Dict[NodeKey, float]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links] # type: List[float]
            s = aggregation(node_inputs) # type: float
            ovalues[node] = activation(bias + response * s)

        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config): # type: (DefaultGenome, Config) -> RecurrentNetwork
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        genome_config = config.genome_config # type: DefaultGenomeConfig
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {} # type: Dict[NodeKey, List[Tuple[NodeKey, float]]]
        for cg in itervalues(genome.connections):
            if not cg.enabled:
                continue

            i, o = cast(ConnKey,cg.key) # type: NodeKey, NodeKey
            if o not in required and i not in required:
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = [] # type: List[Tuple[NodeKey, NormActFunc, NormAgFunc, float, float, List[Tuple[NodeKey, float]]]]
        for node_key, inputs in iteritems(node_inputs):
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation) # type: ignore
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation) # type: ignore
            node_evals.append((node_key,
                               cast(NormActFunc,activation_function),
                               cast(NormAgFunc,aggregation_function),
                               node.bias, node.response, # type: ignore
                               inputs))

        return RecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)
