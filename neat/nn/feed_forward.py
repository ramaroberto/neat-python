from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from neat.mypy_util import DefaultGenome, Config # pylint: disable=unused-import
    from neat.genes import DefaultConnectionGene # pylint: disable=unused-import
    from neat.multiparameter import NormActFunc, NormAgFunc # pylint: disable=unused-import

class FeedForwardNetwork(object):
    def __init__(self,
                 inputs, # type: List[NodeKey]
                 outputs, # type: List[NodeKey]
                 node_evals # type: List[Tuple[NodeKey, NormActFunc, NormAgFunc, float, float, List[Tuple[NodeKey, float]]]]
                 ):
        # type: (...) -> None
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs) # type: Dict[NodeKey, float]

    def activate(self, inputs): # type: (List[float]) -> List[float]
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = [] # type: List[float]
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs) # type: float
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config): # type: (DefaultGenome, Config) -> FeedForwardNetwork
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cast(ConnKey,cg.key) for cg in itervalues(genome.connections) if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = [] # List[Tuple[NodeKey, ActFunc, AgFunc, float, float, List[Tuple[NodeKey, float]]]]
        for layer in layers:
            for node in layer:
                inputs = [] # type: List[Tuple[NodeKey, float]]
                node_expr = [] # type: List[str] # unused?
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight)) # type: ignore
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight)) # type: ignore

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation) # type: ignore
                activation_function = config.genome_config.activation_defs.get(ng.activation) # type: ignore
                node_evals.append((node,
                                   cast(NormActFunc,activation_function),
                                   cast(NormAgFunc,aggregation_function),
                                   ng.bias, ng.response, # type: ignore
                                   inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


