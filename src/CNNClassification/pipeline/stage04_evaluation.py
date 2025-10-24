from src.CNNClassification.config import ConfigurationManager
from src.CNNClassification.components import Evaluation
from src.CNNClassification import logger

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()
            