import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from multimodal_machine_learning.predictor.recurrence import (
    TabularMergedRecurrencePredictorInDistribution, TabularMergedRecurrencePredictorOropharynx,
    TabularMergedRecurrencePredictorOutDistribution
)
from model_evaluation.model_evaluator import TabularPredictorComparer
from model_evaluation.model_evaluator import ModelComparisonMetricsToLatexTable


# ===================================================================================================================
# Models with optimal hyperparameters from the hyperparameter search
# ====================================================================================================================
# ===================================================================================================================
## Random Forest
# ====================================================================================================================
class OptimalRandomForestTabularRecurrencePredictorInDistribution(
    TabularMergedRecurrencePredictorInDistribution
):
    """
    Optimal Random Forest model for predicting recurrence on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=1600, min_samples_split=2,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features='log2', max_depth=30,
            criterion='gini',
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class OptimalRandomForestTabularRecurrencePredictorOropharynx(
    TabularMergedRecurrencePredictorOropharynx
):
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            random_state=np.random.RandomState(self._random_number),
            n_estimators=1200,
            min_samples_split=2,
            min_samples_leaf=1,
            max_leaf_nodes=1000,
            max_features='sqrt',
            max_depth=20,
            criterion='gini'
        )
        return model


class OptimalRandomForestTabularRecurrencePredictorOutDistribution(
    TabularMergedRecurrencePredictorOutDistribution
):
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            random_state=np.random.RandomState(self._random_number),
            n_estimators=800,
            min_samples_split=2,
            min_samples_leaf=1,
            max_leaf_nodes=100,
            max_features='sqrt',
            max_depth=80,
            criterion='log_loss'
        )
        return model


# ===================================================================================================================
## Support Vector Classifier
# ====================================================================================================================
class OptimalSVCTabularRecurrencePredictorInDistribution(
    TabularMergedRecurrencePredictorInDistribution
):
    """
    Optimal Support Vector Classifier model for predicting recurrence on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> SVC:
        model = SVC(
            max_iter=10000,
            kernel='rbf',
            gamma=0.1,
            degree=7,
            C=1000,
            probability=True,
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class OptimalSVCTabularRecurrencePredictorOropharynx(
    TabularMergedRecurrencePredictorOropharynx
):
    def _create_new_model(self) -> SVC:
        model = SVC(
            random_state=np.random.RandomState(self._random_number),
            max_iter=10000,
            kernel='rbf',
            gamma=0.1,
            degree=7,
            C=1000,
            probability=True
        )
        return model


class OptimalSVCTabularRecurrencePredictorOutDistribution(
    TabularMergedRecurrencePredictorOutDistribution
):
    def _create_new_model(self) -> SVC:
        model = SVC(
            random_state=np.random.RandomState(self._random_number),
            max_iter=10000,
            kernel='rbf',
            gamma=0.1,
            degree=7,
            C=1000,
            probability=True
        )
        return model


# ===================================================================================================================
## AdaBoost
# ====================================================================================================================
class OptimalAdaBoostTabularRecurrencePredictorInDistribution(
    TabularMergedRecurrencePredictorInDistribution
):
    """
    Optimal Ada Boost model for predicting recurrence on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and random search.
    """
    def _create_new_model(self) -> AdaBoostClassifier:
        estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_number),
            max_depth=3,
            criterion='gini',
            min_samples_leaf=1,
            min_samples_split=2,
            splitter='best'
        )
        model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_number),
            estimator=estimator,
            n_estimators=1000,
            learning_rate=1.0,
            algorithm='SAMME'
        )
        return model


class OptimalAdaBoostTabularRecurrencePredictorOropharynx(
    TabularMergedRecurrencePredictorOropharynx
):
    def _create_new_model(self) -> AdaBoostClassifier:
        estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_number),
            criterion='log_loss',
            max_depth=3,
            min_samples_leaf=1,
            min_samples_split=2,
            splitter='best'
        )
        model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_number),
            estimator=estimator,
            learning_rate=1.0,
            n_estimators=1000
        )
        return model


class OptimalAdaBoostTabularRecurrencePredictorOutDistribution(
    TabularMergedRecurrencePredictorOutDistribution
):
    def _create_new_model(self) -> AdaBoostClassifier:
        estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_number),
            criterion='gini',
            max_depth=3,
            min_samples_leaf=1,
            min_samples_split=2,
            splitter='best'
        )
        model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_number),
            estimator=estimator,
            learning_rate=1.0,
            n_estimators=1000
        )
        return model


# ===================================================================================================================
# Comparison of optimal Models
# ====================================================================================================================
def compare_recurrence_models(
        distribution: str = 'In', plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
):
    """
    Compare the optimal models for the recurrence prediction on the tabular data.

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
            OptimalRandomForestTabularRecurrencePredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            TabularMergedRecurrencePredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalAdaBoostTabularRecurrencePredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalSVCTabularRecurrencePredictorInDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            )
        ]
    elif distribution == 'Out':
        predictor_list = [
            OptimalRandomForestTabularRecurrencePredictorOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            TabularMergedRecurrencePredictorOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalAdaBoostTabularRecurrencePredictorOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalSVCTabularRecurrencePredictorOutDistribution(
                save_flag=False, plot_flag=False,
                random_state=random_state
            )
        ]
    else:
        predictor_list = [
            OptimalRandomForestTabularRecurrencePredictorOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            TabularMergedRecurrencePredictorOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalAdaBoostTabularRecurrencePredictorOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            ),
            OptimalSVCTabularRecurrencePredictorOropharynx(
                save_flag=False, plot_flag=False,
                random_state=random_state
            )
        ]
    predictor_names = ['Random Forest', 'Random Forest Small', 'Ada Boost', 'Support Vector Classifier']
    predictor_short_names = ['RF', 'RF S', 'Ada', 'SVC']
    plot_name = f'comparison_recurrence_{distribution}_tabular_merged'
    comparer = TabularPredictorComparer(
        predictor_list=predictor_list, predictor_names=predictor_names,
        predictor_short_names=predictor_short_names,
        save_flag=save_flag, plot_flag=plot_flag, random_state=random_state,
        plot_name=plot_name
    )
    return comparer.compare_models()


# ===================================================================================================================
# Execution
# ====================================================================================================================
if __name__ == '__main__':
    main_save_flag = False
    main_plot_flag = False
    main_print_latex_table = False
    metrics_in = compare_recurrence_models(
        distribution='In', plot_flag=main_plot_flag, save_flag=main_save_flag, random_state=42
    )
    metrics_out = compare_recurrence_models(
        distribution='Out', plot_flag=main_plot_flag, save_flag=main_save_flag, random_state=42
    )
    metrics_oropharynx = compare_recurrence_models(
        distribution='Oropharynx', plot_flag=main_plot_flag, save_flag=main_save_flag, random_state=42
    )
    metrics_table = ModelComparisonMetricsToLatexTable(
        metrics=[metrics_in, metrics_out, metrics_oropharynx], data_splits=['In', 'Out', 'Oropharynx'],
        prediction_target='Recurrence Tabular'
    )
    latex_table = metrics_table.create_latex_table_string()
    if main_print_latex_table:
        print(latex_table)

