"""Handles the continuous-time recurrent neural network implementation."""
from __future__ import division

from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
from neat.genes import DefaultNodeGene # pylint: disable=unused-import

from neat.mypy_util import * # pylint: disable=unused-wildcard-import

if MYPY: # pragma: no cover
    from neat.mypy_util import DefaultGenome, DefaultGenomeConfig, Config # pylint: disable=unused-import
    from neat.multiparameter import NormActFunc, NormAgFunc # pylint: disable=unused-import

class CTRNNNodeEval(object):
    def __init__(self,
                 time_constant, # type: float
                 activation, # type: NormActFunc
                 aggregation, # type: NormAgFunc
                 bias, # type: float
                 response, # type: float
                 links # type: List[Tuple[NodeKey, float]]
                 ):
        # type: (...) -> None
        self.time_constant = time_constant
        self.activation = activation
        self.aggregation = aggregation
        self.bias = bias
        self.response = response
        self.links = links


class CTRNN(object):
    def __init__(self,
                 inputs, # type: List[NodeKey]
                 outputs, # type: List[NodeKey]
                 node_evals # type: Dict[NodeKey, CTRNNNodeEval]
                 ):
        # type: (...) -> None
        """Sets up the ctrnn network itself."""
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}] # type: List[Dict[NodeKey, float]]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, ne in iteritems(self.node_evals):
                v[node] = 0.0
                for i, w in ne.links:
                    v[i] = 0.0

        self.active = 0 # type: int # c_type: c_uint
        self.time_seconds = 0.0 # type: float

    def reset(self): # type: () -> None
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0
        self.time_seconds = 0.0

    def set_node_value(self, node_key, value): # type: (NodeKey, float) -> None
        for v in self.values:
            v[node_key] = value

    def get_max_time_step(self): # pragma: no cover
        # TODO: Compute max time step that is known to be numerically stable for
        # the current network configuration.
        # pylint: disable=no-self-use
        raise NotImplementedError()

    def advance(self,
                inputs, # type: List[float]
                advance_time, # type: float
                time_step=None # type: Optional[float]
                ):
        # type: (...) -> List[float]
        """
        Advance the simulation by the given amount of time, assuming that inputs are
        constant at the given values during the simulated time.
        """
        final_time_seconds = self.time_seconds + advance_time

        # Use half of the max allowed time step if none is given.
        if time_step is None: # pragma: no cover
            time_step = 0.5 * self.get_max_time_step()

        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))

        while self.time_seconds < final_time_seconds:
            dt = min(time_step, final_time_seconds - self.time_seconds) # type: float

            ivalues = self.values[self.active]
            ovalues = self.values[1 - self.active]
            self.active = 1 - self.active

            for i, v in zip(self.input_nodes, inputs):
                ivalues[i] = v
                ovalues[i] = v

            for node_key, ne in iteritems(self.node_evals):
                node_inputs = [ivalues[i] * w for i, w in ne.links] # type: List[float]
                s = ne.aggregation(node_inputs) # type: float
                z = ne.activation(ne.bias + ne.response * s) # type: float
                ovalues[node_key] += dt / ne.time_constant * (-ovalues[node_key] + z)

            self.time_seconds += dt

        ovalues = self.values[1 - self.active]
        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, # type: DefaultGenome
               config, # type: Config
               time_constant # type: float
               ):
        # type: (...) -> CTRNN
        """ Receives a genome and returns its phenotype (a CTRNN). """
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

        node_evals = {} # type: Dict[NodeKey, CTRNNNodeEval]
        for node_key, inputs in iteritems(node_inputs):
            node = cast(DefaultNodeGene,genome.nodes[node_key])
            activation_function = genome_config.activation_defs.get(node.activation) # type: ignore
            aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation) # type: ignore
            node_evals[node_key] = CTRNNNodeEval(time_constant,
                                                 cast(NormActFunc,activation_function),
                                                 cast(NormAgFunc,aggregation_function),
                                                 node.bias, # type: ignore
                                                 node.response, # type: ignore
                                                 inputs)

        return CTRNN(genome_config.input_keys, genome_config.output_keys, node_evals)
