import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from multimodal_machine_learning.predictor.adjuvant_treatment import TabularMergedAdjuvantTreatmentPredictor
from multimodal_machine_learning.predictor.adjuvant_treatment import TabularMergedAttentionMlpAdjuvantTreatmentPredictor
from model_evaluation.model_evaluator import TabularPredictorComparer


# ===================================================================================================================
# Models with optimal hyperparameters from the hyperparameter search
# ====================================================================================================================
class OptimalRandomForestTabularAdjuvantTreatmentPredictor(
    TabularMergedAdjuvantTreatmentPredictor
):
    """
    Optimal Random Forest model for predicting adjuvant therapy on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=600, min_samples_split=5,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features='sqrt', max_depth=50,
            criterion='log_loss',
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class OptimalSVCTabularAdjuvantTreatmentPredictor(
    TabularMergedAdjuvantTreatmentPredictor
):
    """
    Optimal Support Vector Classifier model for predicting adjuvant therapy on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> SVC:
        model = SVC(
            max_iter=1000,
            kernel='rbf',
            gamma=1,
            degree=10,
            C=10,
            probability=True,
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class OptimalAdaBoostTabularAdjuvantTreatmentPredictor(
    TabularMergedAdjuvantTreatmentPredictor
):
    """
    Optimal Ada Boost model for predicting adjuvant therapy on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> AdaBoostClassifier:
        estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_number),
            max_depth=3,
            criterion='gini'
        )
        model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_number),
            estimator=estimator,
            n_estimators=1000,
            learning_rate=0.1,
            algorithm='SAMME'
        )
        return model


# ===================================================================================================================
# Comparison between optimal models
# ====================================================================================================================
class TabularAdjuvantTherapyComparer(
    TabularPredictorComparer
):
    def __init__(
            self, save_flag: bool = False,
            plot_flag: bool = False, random_state: int = 42
    ):
        """Creates a comparer for the different optimal models for the adjuvant therapy predictors.
        To change the predictors that should be used, change the _predictors list together with the
        _predictor_names and _predictor_short_names list.

        Args:
            save_flag (bool, optional): If the results should be saved. Defaults to False.
            plot_flag (bool, optional): If the results should be plotted. Defaults to False
            random_state (int, optional): The random state for the models. Defaults to 42.
        """
        predictor_list = [
            OptimalRandomForestTabularAdjuvantTreatmentPredictor(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalAdaBoostTabularAdjuvantTreatmentPredictor(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalSVCTabularAdjuvantTreatmentPredictor(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            TabularMergedAttentionMlpAdjuvantTreatmentPredictor(
                save_flag=False, plot_flag=False,
                random_state=random_state
            )
        ]
        predictor_names = ['Random Forest', 'Ada Boost', 'Support Vector Classifier', 'MLP']
        predictor_short_names = ['RF', 'Ada', 'SVC', 'MLP']
        super().__init__(
            predictor_list=predictor_list, predictor_names=predictor_names, predictor_short_names=predictor_short_names,
            save_flag=save_flag, plot_flag=plot_flag, random_state=random_state,
            plot_name='comparison_adjuvant_therapy_tabular_merged'
        )


if __name__ == '__main__':
    comparer = TabularAdjuvantTherapyComparer(
        save_flag=True, plot_flag=True, random_state=42
    )
    comparer.compare_models()
