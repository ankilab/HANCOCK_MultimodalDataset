# ===================================================================================================================
# Imports
# ====================================================================================================================
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from multimodal_machine_learning.predictor.recurrence import (
    TabularMergedRecurrencePredictorInDistribution,
    TabularMergedRecurrencePredictorOutDistribution,
    TabularMergedRecurrencePredictorOropharynx
)
from model_evaluation.model_evaluator import (
    SvcPredictorModelEvaluator,
    RandomForestPredictorModelEvaluator,
    AdaBoostPredictorModelEvaluator,
    LogisticRegressorPredictorModelEvaluator
)


# ===================================================================================================================
# Recurrence Hyperparameter Evaluator
# ====================================================================================================================
class RandomForestRecurrenceInDistributionModelEvaluator(
    RandomForestPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorInDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_random_forest_recurrence_in_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class SVCTabularRecurrenceInDistributionModelEvaluator(
    SvcPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorInDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_svc_recurrence_in_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class AdaBoostTabularRecurrenceInDistributionModelEvaluation(
    AdaBoostPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorInDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_ada_boost_recurrence_in_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class RandomForestRecurrenceOutDistributionModelEvaluator(
    RandomForestPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorOutDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_random_forest_recurrence_out_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class SVCTabularRecurrenceOutDistributionModelEvaluator(
    SvcPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorOutDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_svc_recurrence_out_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class AdaBoostTabularRecurrenceOutDistributionModelEvaluation(
    AdaBoostPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorOutDistribution,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_ada_boost_recurrence_out_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class RandomForestRecurrenceOropharynxDistributionModelEvaluator(
    RandomForestPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorOropharynx,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_random_forest_recurrence_Oropharynx_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class SVCTabularRecurrenceOropharynxDistributionModelEvaluator(
    SvcPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorOropharynx,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_svc_recurrence_Oropharynx_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


class AdaBoostTabularRecurrenceOropharynxDistributionModelEvaluation(
    AdaBoostPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int =42
    ):
        super().__init__(
            predictor_cls=TabularMergedRecurrencePredictorOropharynx,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_ada_boost_recurrence_Oropharynx_distribution'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring, plot_name=plot_name
        )


def random_search_recurrence(
        distribution: str = 'In', plot_flag: bool = False,
        save_flag: bool = False, n_splits: int = 10,
        n_iter: int = 100
):
    if distribution not in ['In', 'Out', 'Oropharynx']:
        raise ValueError(f'Invalid distribution: {distribution}',
                         'Valid distributions are: In, Out, Oropharynx')

    if distribution == 'In':
        print('Random Forest Recurrence In Model Evaluation')
        random_forest_evaluator = RandomForestRecurrenceInDistributionModelEvaluator(
            plot_flag=plot_flag, save_flag=save_flag
        )
        random_forest_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )

        print('SVC Recurrence In Model Evaluation')
        svc_evaluator = SVCTabularRecurrenceInDistributionModelEvaluator(
            plot_flag=plot_flag, save_flag=save_flag
        )
        svc_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )

        print('AdaBoost Recurrence In Model Evaluation')
        ada_boost_evaluator = AdaBoostTabularRecurrenceInDistributionModelEvaluation(
            plot_flag=plot_flag, save_flag=save_flag
        )
        ada_boost_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )

    elif distribution == 'Out':
        print('Random Forest Recurrence Out Model Evaluation')
        random_forest_evaluator = RandomForestRecurrenceOutDistributionModelEvaluator(
            plot_flag=plot_flag, save_flag=save_flag
        )
        random_forest_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )

        print('SVC Recurrence Out Model Evaluation')
        svc_evaluator = SVCTabularRecurrenceOutDistributionModelEvaluator(
            plot_flag=plot_flag, save_flag=save_flag
        )
        svc_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )

        print('AdaBoost Recurrence Out Model Evaluation')
        ada_boost_evaluator = AdaBoostTabularRecurrenceOutDistributionModelEvaluation(
            plot_flag=plot_flag, save_flag=save_flag
        )
        ada_boost_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )
    elif distribution == 'Oropharynx':
        print('Random Forest Recurrence Oropharynx Model Evaluation')
        random_forest_evaluator = RandomForestRecurrenceOropharynxDistributionModelEvaluator(
            plot_flag=plot_flag, save_flag=save_flag
        )
        random_forest_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )

        print('SVC Recurrence Oropharynx Model Evaluation')
        svc_evaluator = SVCTabularRecurrenceOropharynxDistributionModelEvaluator(
            plot_flag=plot_flag, save_flag=save_flag
        )
        svc_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )

        print('AdaBoost Recurrent Oropharynx Model Evaluation')
        ada_boost_evaluator = AdaBoostTabularRecurrenceOropharynxDistributionModelEvaluation(
            plot_flag=plot_flag, save_flag=save_flag
        )
        ada_boost_evaluator.random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter
        )


if __name__ == '__main__':
    main_plot_flag = True
    main_save_flag = True
    main_n_splits = 10
    main_n_iter = 100
    do_it_flag = True

    if do_it_flag:
        random_search_recurrence(
            distribution='In', plot_flag=main_plot_flag, save_flag=main_save_flag,
            n_splits=main_n_splits, n_iter=main_n_iter
        )
        random_search_recurrence(
            distribution='Out', plot_flag=main_plot_flag, save_flag=main_save_flag,
            n_splits=main_n_splits, n_iter=main_n_iter
        )
        random_search_recurrence(
            distribution='Oropharynx', plot_flag=main_plot_flag, save_flag=main_save_flag,
            n_splits=main_n_splits, n_iter=main_n_iter
        )

