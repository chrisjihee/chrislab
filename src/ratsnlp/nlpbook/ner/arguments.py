from dataclasses import dataclass

from ratsnlp.nlpbook.arguments import NLUArguments, NLUTrainerArguments, NLUServerArguments


@dataclass
class NERArguments(NLUArguments):
    pass


@dataclass
class NERTrainArguments(NLUTrainerArguments):
    pass


@dataclass
class NERDeployArguments(NLUServerArguments):
    pass
