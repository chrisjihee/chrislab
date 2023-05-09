from dataclasses import dataclass

from ratsnlp.nlpbook.arguments import NLUArguments, NLUTrainArguments, NLUDeployArguments


@dataclass
class ClassificationArguments(NLUArguments):
    pass


@dataclass
class ClassificationTrainArguments(NLUTrainArguments):
    pass


@dataclass
class ClassificationDeployArguments(NLUDeployArguments):
    pass
