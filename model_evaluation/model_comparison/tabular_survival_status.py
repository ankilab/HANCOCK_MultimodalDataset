import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning.predictor.survival_status import (
    TabularMergedSurvivalStatusPredictorInDistribution, TabularMergedSurvivalStatusPredictorOropharynx,
    TabularMergedSurvivalStatusPredictorOutDistribution
)
from model_evaluation.model_evaluator import TabularPredictorComparer
from model_evaluation.model_evaluator import ModelComparisonMetricsToLatexTable


# ===================================================================================================================
# Models with optimal hyperparameters from the hyperparameter search
# ====================================================================================================================
# ===================================================================================================================
## Random Forest Optimal
# ====================================================================================================================
class OptimalRandomForestTabularSurvivalPredictorInDistribution(
    TabularMergedSurvivalStatusPredictorInDistribution
):
    """
    Optimal Random Forest model for predicting survival on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=1000, min_samples_split=5,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features='log2', criterion='entropy',
            # max_depth=null,
            random_state=np.random.RandomState(self._random_number)
        )
        return model

class OptimalRandomForestTabularSurvivalPredictorOropharynx(
    TabularMergedSurvivalStatusPredictorOropharynx
):
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=1200, min_samples_split=2,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features='sqrt', max_depth=20,
            criterion='gini',
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class OptimalRandomForestTabularSurvivalPredictorOutDistribution(
    TabularMergedSurvivalStatusPredictorOutDistribution
):
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            random_state=np.random.RandomState(self._random_number),
            n_estimators=1200, min_samples_split=2,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features='sqrt', max_depth=20,
            criterion='gini'
        )
        return model


# ===================================================================================================================
## Support Vector Classifier
# ====================================================================================================================
class OptimalSVCTabularSurvivalPredictorInDistribution(
    TabularMergedSurvivalStatusPredictorInDistribution
):
    """
    Optimal Support Vector Classifier model for predicting survival status on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> SVC:
        model = SVC(
            max_iter=10000,
            kernel='rbf',
            gamma=0.1,
            degree=7,
            C=10,
            probability=True,
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class OptimalSVCTabularSurvivalPredictorOropharynx(
    TabularMergedSurvivalStatusPredictorOropharynx
):
    def _create_new_model(self) -> SVC:
        model = SVC(
            random_state=np.random.RandomState(self._random_number),
            probability = True,
            max_iter=10000, kernel='rbf',
            gamma=0.1, degree=7, C=1000
        )
        return model


class OptimalSVCTabularSurvivalPredictorOutDistribution(
    TabularMergedSurvivalStatusPredictorOutDistribution
):
    def _create_new_model(self) -> SVC:
        model = SVC(
            random_state=np.random.RandomState(self._random_number),
            probability = True,
            max_iter=10000, kernel='rbf',
            gamma=0.1, degree=7, C=1000
        )
        return model


# ===================================================================================================================
## AdaBoost
# ====================================================================================================================
class OptimalAdaBoostTabularSurvivalPredictorInDistribution(
    TabularMergedSurvivalStatusPredictorInDistribution
):
    """
    Optimal Ada Boost model for predicting survival status on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """

    def __init__(
            self, save_flag: bool = False, plot_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            save_flag=save_flag,
            plot_flag=plot_flag,
            random_state=random_state
        )

    def _create_new_model(self) -> AdaBoostClassifier:
        estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_number),
            max_depth=3, min_samples_leaf=1, min_samples_split=2,
            splitter='best',
            criterion='log_loss'
        )
        model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_number),
            estimator=estimator,
            n_estimators=1000,
            learning_rate=0.1,
            algorithm='SAMME'
        )
        return model


class OptimalAdaBoostTabularSurvivalPredictionOropharynx(
    TabularMergedSurvivalStatusPredictorOropharynx
):
    def _create_new_model(self) -> AdaBoostClassifier:
        estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_number),
            criterion='gini', max_depth=3, min_samples_split=2, min_samples_leaf=1,
            splitter='best'
        )
        model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_number),
            estimator=estimator, learning_rate=1.0, n_estimators=500
        )
        return model


class OptimalAdaBoostTabularSurvivalPredictionOutDistribution(
    TabularMergedSurvivalStatusPredictorOutDistribution
):
    def _create_new_model(self) -> AdaBoostClassifier:
        estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_number),
            criterion='log_loss', max_depth=3, min_samples_leaf=1, min_samples_split=2,
            splitter='best'
        )
        model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_number),
            estimator=estimator, learning_rate=1.0, n_estimators=1000
        )
        return model


# ===================================================================================================================
# Comparison of optimal Models
# ====================================================================================================================
def compare_survival_models(
        distribution: str = 'In', plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
):
    """
    Compare the optimal models for the survival prediction on the tabular data.

    Args:
        distribution (str, optional): The distribution of the data. Defaults to 'In'.
            Valid values are: 'In', 'Out', 'Oropharynx'.
        plot_flag (bool, optional): If the results should be plotted set this to True.
            Defaults to False.
        save_flag (bool, optional): If the results should be saved set this to True.
            Either the plot_flag or the save_flag should be set to True, otherwise only computation power is wasted.
            Defaults to False.
        random_state (int, optional): The random state for the models. Defaults to 42.
    """
    if distribution not in ['In', 'Out', 'Oropharynx']:
        raise ValueError('Invalid distribution type: {}'.format(distribution))

    if distribution == 'In':
        predictor_list = [
            OptimalRandomForestTabularSurvivalPredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            TabularMergedSurvivalStatusPredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalAdaBoostTabularSurvivalPredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalSVCTabularSurvivalPredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            )
        ]
    elif distribution == 'Out':
        predictor_list = [
            OptimalRandomForestTabularSurvivalPredictorOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            TabularMergedSurvivalStatusPredictorOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalAdaBoostTabularSurvivalPredictionOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalSVCTabularSurvivalPredictorOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            )
        ]
    else:
        predictor_list = [
            OptimalRandomForestTabularSurvivalPredictorOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            TabularMergedSurvivalStatusPredictorOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalAdaBoostTabularSurvivalPredictionOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalSVCTabularSurvivalPredictorOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            )
        ]
    predictor_names = ['Random Forest', 'Random Forest Small', 'Ada Boost', 'Support Vector Classifier']
    predictor_short_names = ['RF', 'RF S', 'Ada', 'SVC']
    plot_name = f'comparison_survival_{distribution}_tabular_merged'
    comparer = TabularPredictorComparer(
        predictor_list=predictor_list, predictor_names=predictor_names,
        predictor_short_names=predictor_short_names,
        save_flag=save_flag, plot_flag=plot_flag, random_state=random_state,
        plot_name=plot_name
    )
    return comparer.compare_models()


if __name__ == '__main__':
    main_save_flag = False
    main_plot_flag = False
    main_print_latex_table = True
    metrics_in = compare_survival_models(
        distribution='In', plot_flag=main_plot_flag, save_flag=main_save_flag, random_state=42
    )
    metrics_out = compare_survival_models(
        distribution='Out', plot_flag=main_plot_flag, save_flag=main_save_flag, random_state=42
    )
    metrics_oropharynx = compare_survival_models(
        distribution='Oropharynx', plot_flag=main_plot_flag, save_flag=main_save_flag, random_state=42
    )
    metrics_table = ModelComparisonMetricsToLatexTable(
        metrics=[metrics_in, metrics_out, metrics_oropharynx], data_splits=['In', 'Out', 'Oropharynx'],
        prediction_target='Survival Tabular'
    )
    latex_table = metrics_table.create_latex_table_string()
    if main_print_latex_table:
        print(latex_table)
