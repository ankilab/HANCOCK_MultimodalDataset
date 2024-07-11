import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning import TabularAdjuvantTreatmentPredictor


class TabularAdjuvantTreatmentModelEvaluator:
    def __init__(self):
        plot_flag = True
        save_flag = False
        self.predictor = TabularAdjuvantTreatmentPredictor(
            plot_flag=plot_flag, save_flag=save_flag
        )

    def evaluate_with_cross_validation(self):


