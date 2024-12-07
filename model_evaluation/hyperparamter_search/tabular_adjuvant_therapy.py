# ===================================================================================================================
# Imports
# ====================================================================================================================
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))
from multimodal_machine_learning.predictor.adjuvant_treatment import TabularMergedAdjuvantTreatmentPredictor
from model_evaluation.model_evaluator import (
    SvcPredictorModelEvaluator,
    RandomForestPredictorModelEvaluator,
    AdaBoostPredictorModelEvaluator,
    LogisticRegressorPredictorModelEvaluator
)


class RandomForestTabularAdjuvantTreatmentModelEvaluator(
    RandomForestPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        """Creates an instance of RandomForestTabularAdjuvantTreatmentModelEvaluator.

        Args:
            plot_flag (bool, optional): Flag to plot the results. Defaults to False.
            save_flag (bool, optional): Flag to save the results. Defaults to False
            random_state (int, optional): Random state seed for reproducibility. Will be passed to
            all functions classes that contain random events. Defaults to 42.
        """
        super().__init__(
            predictor_cls = TabularMergedAdjuvantTreatmentPredictor,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100,
            scoring: str = 'f1',
            plot_name: str = 'random_search_random_forest_adjuvant_therapy_multimodal'
    ) -> dict[str, any]:
        """Performs a random search with cross-validation on the model
        (call of _create_base_model()) using the hyperparameter space
        defined (call of create_hyper_parameter_space_random()).

        Args:
            n_splits (int, optional): Number of splits for cross-validation.
            Defaults to 10.

            n_iter (int, optional): Number of iterations for the random search.
            Will choose n_iter hyperparameter combinations. Defaults to 100.

            scoring (str, optional): Scoring metric for the selection of the best model.
            For more metrics check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.
            Defaults to 'f1'.

            plot_name (str, optional): Name of the plot (also used for saving the plot).
            Defaults to 'random_search_adjuvant_therapy'.

        Returns:
            dict[str, any]: Returns a dictionary with the key 'train_returns' and the value
            of the training returns of the model.
            The train_returns are:
            list: A list with the validation parameters and the data that was used.
            [[fpr, tpr, auc], [x_train, y_train, x_other, y_other, y_pred], features]
            fpr (np.array): False positive rate
            tpr (np.array): True positive rate
            auc (float): The AUC score
            x_train (np.ndarray): The training data encoded
            y_train (np.array): The training labels
            x_other (np.ndarray): The validation data encoded
            y_other (np.array): The validation labels
            y_pred (np.array): The predicted labels for the validation data
            features (list): The features from the ColumnTransformer that was used for
            encoding the training data.
        """
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring,
            plot_name=plot_name
        )


class SVCTabularAdjuvantTreatmentModelEvaluator(
    SvcPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = True,
            random_state: int = 42
    ) -> None:
        """Creates an instance of SVCTabularAdjuvantTreatmentModelEvaluator.

        Args:
            plot_flag (bool, optional): Flag to plot the results. Defaults to False.
            save_flag (bool, optional): Flag to save the results. Defaults to False
            random_state (int, optional): Random state seed for reproducibility. Will be passed to
            all functions classes that contain random events. Defaults to 42.
        """
        super().__init__(
            predictor_cls = TabularMergedAdjuvantTreatmentPredictor,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_svc_adjuvant_therapy'
    ) -> dict[str, any]:
        """Performs a random search with cross-validation on the model
        (call of _create_base_model()) using the hyperparameter space
        defined (call of create_hyper_parameter_space_random()).

        Args:
            n_splits (int, optional): Number of splits for cross-validation.
            Defaults to 10.

            n_iter (int, optional): Number of iterations for the random search.
            Will choose n_iter hyperparameter combinations. Defaults to 100.

            scoring (str, optional): Scoring metric for the selection of the best model.
            For more metrics check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.
            Defaults to 'f1'.

            plot_name (str, optional): Name of the plot (also used for saving the plot).
            Defaults to 'random_search_adjuvant_therapy'.

        Returns:
            dict[str, any]: Returns a dictionary with the key 'train_returns' and the value
            of the training returns of the model.
            The train_returns are:
            list: A list with the validation parameters and the data that was used.
            [[fpr, tpr, auc], [x_train, y_train, x_other, y_other, y_pred], features]
            fpr (np.array): False positive rate
            tpr (np.array): True positive rate
            auc (float): The AUC score
            x_train (np.ndarray): The training data encoded
            y_train (np.array): The training labels
            x_other (np.ndarray): The validation data encoded
            y_other (np.array): The validation labels
            y_pred (np.array): The predicted labels for the validation data
            features (list): The features from the ColumnTransformer that was used for
            encoding the training data.
        """
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring,
            plot_name=plot_name
        )


class AdaBoostTabularAdjuvantTreatmentModelEvaluator(
    AdaBoostPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        """Creates an instance of AdaBoostTabularAdjuvantTreatmentModelEvaluator.

        Args:
            plot_flag (bool, optional): Flag to plot the results. Defaults to False.
            save_flag (bool, optional): Flag to save the results. Defaults to False
            random_state (int, optional): Random state seed for reproducibility. Will be passed to
            all functions classes that contain random events. Defaults to 42.
        """
        super().__init__(
            predictor_cls=TabularMergedAdjuvantTreatmentPredictor,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100,
            scoring: str = 'f1',
            plot_name: str = 'random_search_ada_boost_adjuvant_therapy_multimodal'
    ) -> dict[str, any]:
        """Performs a random search with cross-validation on the model
        (call of _create_base_model()) using the hyperparameter space
        defined (call of create_hyper_parameter_space_random()).

        Args:
            n_splits (int, optional): Number of splits for cross-validation.
            Defaults to 10.

            n_iter (int, optional): Number of iterations for the random search.
            Will choose n_iter hyperparameter combinations. Defaults to 100.

            scoring (str, optional): Scoring metric for the selection of the best model.
            For more metrics check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.
            Defaults to 'f1'.

            plot_name (str, optional): Name of the plot (also used for saving the plot).
            Defaults to 'random_search_adjuvant_therapy'.

        Returns:
            dict[str, any]: Returns a dictionary with the key 'train_returns' and the value
            of the training returns of the model.
            The train_returns are:
            list: A list with the validation parameters and the data that was used.
            [[fpr, tpr, auc], [x_train, y_train, x_other, y_other, y_pred], features]
            fpr (np.array): False positive rate
            tpr (np.array): True positive rate
            auc (float): The AUC score
            x_train (np.ndarray): The training data encoded
            y_train (np.array): The training labels
            x_other (np.ndarray): The validation data encoded
            y_other (np.array): The validation labels
            y_pred (np.array): The predicted labels for the validation data
            features (list): The features from the ColumnTransformer that was used for
            encoding the training data.
        """
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring,
            plot_name=plot_name
        )


class LogRegressorTabularAdjuvantTreatmentModelEvaluator(
    LogisticRegressorPredictorModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        """Creates an instance of LogRegressorTabularAdjuvantTreatmentModelEvaluator.
        Only for the evaluation of the best model for the
        AdaBoostAdjuvantTreatmentModelEvaluator.

        Args:
            plot_flag (bool, optional): Flag to plot the results. Defaults to False.
            save_flag (bool, optional): Flag to save the results. Defaults to False
            random_state (int, optional): Random state seed for reproducibility. Will be passed to
            all functions classes that contain random events. Defaults to 42.
        """
        super().__init__(
            predictor_cls=TabularMergedAdjuvantTreatmentPredictor,
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100,
            scoring: str = 'f1',
            plot_name: str = 'random_search_log_regressor_adjuvant_therapy_multimodal'
    ) -> dict[str, any]:
        """Performs a random search with cross-validation on the model
        (call of _create_base_model()) using the hyperparameter space
        defined (call of create_hyper_parameter_space_random()).

        Args:
            n_splits (int, optional): Number of splits for cross-validation.
            Defaults to 10.

            n_iter (int, optional): Number of iterations for the random search.
            Will choose n_iter hyperparameter combinations. Defaults to 100.

            scoring (str, optional): Scoring metric for the selection of the best model.
            For more metrics check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.
            Defaults to 'f1'.

            plot_name (str, optional): Name of the plot (also used for saving the plot).
            Defaults to 'random_search_adjuvant_therapy'.

        Returns:
            dict[str, any]: Returns a dictionary with the key 'train_returns' and the value
            of the training returns of the model.
            The train_returns are:
            list: A list with the validation parameters and the data that was used.
            [[fpr, tpr, auc], [x_train, y_train, x_other, y_other, y_pred], features]
            fpr (np.array): False positive rate
            tpr (np.array): True positive rate
            auc (float): The AUC score
            x_train (np.ndarray): The training data encoded
            y_train (np.array): The training labels
            x_other (np.ndarray): The validation data encoded
            y_other (np.array): The validation labels
            y_pred (np.array): The predicted labels for the validation data
            features (list): The features from the ColumnTransformer that was used for
            encoding the training data.
        """
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring,
            plot_name=plot_name
        )


if __name__ == "__main__":
    main_plot_flag = True
    main_save_flag = True
    do_it_flag = True
    test_n_splits = 10
    test_n_iter = 100
    if do_it_flag:
        ada_boost_evaluator = AdaBoostTabularAdjuvantTreatmentModelEvaluator(
            plot_flag=main_plot_flag, save_flag=main_save_flag
        )
        _ = ada_boost_evaluator.random_search_with_cross_validation(
            n_splits=test_n_splits, n_iter=test_n_iter
        )

        random_forest_evaluator = RandomForestTabularAdjuvantTreatmentModelEvaluator(
            plot_flag=main_plot_flag, save_flag=main_save_flag
        )
        _ = random_forest_evaluator.random_search_with_cross_validation(
            n_splits=test_n_splits, n_iter=test_n_iter
        )

        svc_evaluator = SVCTabularAdjuvantTreatmentModelEvaluator(
            plot_flag=main_plot_flag, save_flag=main_save_flag
        )
        _ = svc_evaluator.random_search_with_cross_validation(
            n_splits=test_n_splits, n_iter=test_n_iter
        )

        # log_regressor_evaluator = LogRegressorTabularAdjuvantTreatmentModelEvaluator(
        #     plot_flag=main_plot_flag, save_flag=main_save_flag
        # )
        # _ = log_regressor_evaluator.random_search_with_cross_validation(
        #     n_splits=test_n_splits, n_iter=test_n_iter
        # )
