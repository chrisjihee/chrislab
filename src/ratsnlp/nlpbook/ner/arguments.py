from dataclasses import dataclass

from ratsnlp.nlpbook.arguments import NLUArguments, NLUTrainArguments, NLUDeployArguments


@dataclass
class NERArguments(NLUArguments):
    pass


@dataclass
class NERTrainArguments(NLUTrainArguments):
    pass


@dataclass
class NERDeployArguments(NLUDeployArguments):
    pass
