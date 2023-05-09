from dataclasses import dataclass

from ratsnlp.nlpbook.arguments import NLUArguments, NLUTrainerArguments, NLUServerArguments


@dataclass
class ClassificationArguments(NLUArguments):
    pass


@dataclass
class ClassificationTrainArguments(NLUTrainerArguments):
    pass


@dataclass
class ClassificationDeployArguments(NLUServerArguments):
    pass
