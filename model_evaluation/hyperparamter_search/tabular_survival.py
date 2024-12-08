# ===================================================================================================================
# Imports
# ====================================================================================================================
import sys
from pathlib import Path
from random import random

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning.predictor.survival_status import (
    TabularMergedSurvivalStatusPredictorOutDistribution,
    TabularMergedSurvivalStatusPredictorInDistribution,
    TabularMergedSurvivalStatusPredictorOropharynx
)
from model_evaluation.model_evaluator import (
    SvcPredictorModelEvaluator,
    RandomForestPredictorModelEvaluator,
    AdaBoostPredictorModelEvaluator,
    LogisticRegressorPredictorModelEvaluator
)


# ===================================================================================================================
# Survival Status Evaluator
# ====================================================================================================================
class RandomForestSurvivalModelEvaluator(
    RandomForestPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            predictor_cls=TabularMergedSurvivalStatusPredictorInDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_random_forest_survival_status_in_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class SVCTabularSurvivalModelEvaluator(
    SvcPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedSurvivalStatusPredictorInDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_svc_survival_status_in_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class AdaBoostTabularSurvivalModelEvaluation(
    AdaBoostPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedSurvivalStatusPredictorInDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_ada_boost_survival_status_in_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class LogisticRegressorTabularSurvivalModelEvaluator(
    LogisticRegressorPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedSurvivalStatusPredictorInDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_logistic_regressor_survival_status_in_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


# ===================================================================================================================
# Survival Status Hyperparameter Search Execution:
# ====================================================================================================================
def random_search_survival(
        distribution: str = 'In', plot_flag: bool = False, save_flag: bool = False,
        n_splits: int = 10, n_iter: int = 100
):
    if distribution not in ['In', 'Out', 'Oropharynx']:
        raise ValueError('Invalid distribution type: {}'.format(distribution))

    if distribution == 'In':
        predictor_cls = TabularMergedSurvivalStatusPredictorInDistribution
    elif distribution == 'Out':
        predictor_cls = TabularMergedSurvivalStatusPredictorOutDistribution
    else:
        predictor_cls = TabularMergedSurvivalStatusPredictorOropharynx

    print(f'Random Forest Survival Status Model Evaluation for {distribution} Distribution')
    random_forest_evaluator = RandomForestPredictorModelEvaluator(
        predictor_cls=predictor_cls, plot_flag=plot_flag, save_flag=save_flag
    )
    random_forest_evaluator.random_search_with_cross_validation(
        n_splits=n_splits, n_iter=n_iter, scoring='f1',
        plot_name = f'random_search_random_forest_survival_status_{distribution}'
    )

    print(f'SVC Survival Status Model Evaluation for {distribution} Distribution')
    svc_evaluator = SvcPredictorModelEvaluator(
        predictor_cls=predictor_cls, plot_flag=plot_flag, save_flag=save_flag
    )
    svc_evaluator.random_search_with_cross_validation(
        n_splits=n_splits, n_iter=n_iter, scoring='f1',
        plot_name = f'random_search_svc_survival_status_{distribution}'
    )

    print(f'AdaBoost Survival Status Model Evaluation for {distribution} Distribution')
    ada_boost_evaluator = AdaBoostPredictorModelEvaluator(
        predictor_cls=predictor_cls, plot_flag=plot_flag, save_flag=save_flag
    )
    ada_boost_evaluator.random_search_with_cross_validation(
        n_splits=n_splits, n_iter=n_iter, scoring='f1',
        plot_name = f'random_search_ada_boost_survival_status_{distribution}'
    )


if __name__ == '__main__':
    main_plot_flag = True
    main_save_flag = True
    main_n_splits = 10
    main_n_iter = 100
    do_it_flag = True
    if do_it_flag:
        random_search_survival(
            distribution='In', plot_flag=main_plot_flag, save_flag=main_save_flag,
            n_splits=main_n_splits, n_iter=main_n_iter
        )
        random_search_survival(
            distribution='Out', plot_flag=main_plot_flag, save_flag=main_save_flag,
            n_splits=main_n_splits, n_iter=main_n_iter
        )
        random_search_survival(
            distribution='Oropharynx', plot_flag=main_plot_flag, save_flag=main_save_flag,
            n_splits=main_n_splits, n_iter=main_n_iter
        )

