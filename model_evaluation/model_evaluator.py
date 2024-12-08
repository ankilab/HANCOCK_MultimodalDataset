import sys
import pandas as pd
from pathlib import Path
from typing import Type
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import json

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning.predictor import AbstractHancockPredictor
from multimodal_machine_learning.predictor.plotter import PredictionPlotter


# ====================================================================================================================
# Abstract Prediction Model Evaluator
# ====================================================================================================================
class AbstractPredictionModelEvaluator:
    """For Evaluating Hyper-Parameters of a Prediction Model.
    on the merged tabular (multi-modal) data set.

    Methods:
        - random_search_with_cross_validation: Performs a random search with cross-validation on the model.
        - get_best_random_search_model: Returns the best estimator from the random search, but only if the
            random_search_with_cross_validation method was called before.
        - get_best_random_search_parameters: Returns the best parameters from the random search, but only if the
            random_search_with_cross_validation method was called before.

    Abstract Methods:
        - create_hyperparameter_space_random: Should create a hyperparameter space for the base model.
        - _create_base_model: Should return the base model that is used for the evaluation.
    """
    def __init__(
            self,
            predictor_cls: Type[AbstractHancockPredictor],
            plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ) -> None:
        """Creates an instance of the TabularAdjuvantTreatmentModelEvaluator. Should not be used
        directly because this is intended to be an abstract class.

        Args:
            plot_flag (bool, optional): Flag to plot the results. Defaults to False.
            save_flag (bool, optional): Flag to save the results. Defaults to False
            random_state (int, optional): Random state seed for reproducibility. Will be passed to
            all functions classes that contain random events. Defaults to 42.
        """
        self.predictor = predictor_cls(
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )
        self._random_state = random_state
        self._random_search_performed = False
        self._grid_search_performed = False
        self._save_flag = save_flag
        self._plot_flag = plot_flag

        self.hyper_parameter_space_random = self.create_hyperparameter_space_random()
        # self.hyper_parameter_space_grid = self.create_hyper_parameter_space_grid()

    # Random Search
    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_adjuvant_therapy'
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
        base_model = self._create_base_model()
        random_search_predictor = RandomizedSearchCV(
            param_distributions=self.hyper_parameter_space_random,
            estimator=base_model, n_iter=n_iter, cv=n_splits,
            scoring=scoring, verbose=2,
            random_state=np.random.RandomState(self._random_state),
            n_jobs=-1
        )
        self.predictor.model = random_search_predictor
        train_returns = self.predictor.train(plot_name=plot_name)
        self._random_search_performed = True
        self._grid_search_performed = False
        if self._save_flag:
            self._save_random_search_results(file_name=plot_name)
        return {'train_returns': train_returns}

    def get_best_random_search_model(self) -> any:
        """If random search was performed and no grid search inbetween,
        the best estimator will be returned from the random search.

        Returns:
            any: None of no random search was performed, else the best
            estimator.
        """
        if not self._random_search_performed:
            print('Random search not performed!')
            return None
        return self.predictor.model.best_estimator_

    def get_best_random_search_parameters(self) -> dict | None:
        """If random search was performed and no grid search
        inbetween, the best parameters will be returned from the
        random search.

        Returns:
            dict | None: None if the random search was not performed,
            else the best parameters as dictionary.
        """
        if not self._random_search_performed:
            print('Random search not performed!')
            return None
        else:
            return self.predictor.model.best_params_

    # Grid Search
    def grid_search_with_cross_validation(self, n_splits: int = 10):
        raise NotImplementedError('Grid search not implemented!')

    def get_best_grid_search_model(self) -> any:
        if not self._grid_search_performed:
            print('Grid search not performed!')
            return None
        return self.predictor.model.best_estimator_

    def get_grid_search_parameters(self) -> dict | None:
        if not self._grid_search_performed:
            print('Grid search not performed!')
            return None
        return self.predictor.model.best_params_

    # ================================================================================================================
    # Abstract Methods
    # ================================================================================================================
    def create_hyperparameter_space_random(self) -> dict[str, any]:
        """Should create a hyperparameter space for the base model.
        Has to be implemented in the subclass.

        Returns:
            dict[str, any]: A dictionary with the hyperparameter names
            as keys and the values as list of possible values.

        Raises:
            NotImplementedError: If the function is not implemented.
        """
        raise NotImplementedError('Hyperparameter Space Random not implemented!')

    @staticmethod
    def create_hyper_parameter_space_grid() -> dict[str, any]:
        """Should create a hyperparameter space for the base model.
        Has to be implemented in the subclass.

        Returns:
            dict[str, any]: A dictionary with the hyperparameter names
            as keys and the values as list of possible values.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError('Hyperparameter Space Grid not implemented!')

    def _create_base_model(self) -> any:
        """Should return the base model that is used for the evaluation.
        Has to be implemented in the subclass.

        Returns:
            any: The base model that is used for the evaluation.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError('Base model not implemented!')

    # ================================================================================================================
    # Helper Methods
    # ================================================================================================================
    def _save_random_search_results(
            self, file_name: str = 'random_search_adjuvant_therapy'
    ) -> None:
        """If random search was performed the best parameters will be saved
        to a json file in the results directory with the file_name as
        a json file.

        Args:
            file_name (str, optional): Name of the file without file extension.
            Defaults to 'random_search_adjuvant_therapy'.
        """
        best_params = self.get_best_random_search_parameters()
        if best_params is not None:
            save_path = self.predictor.args.results_dir / f'{file_name}.json'
            with open(save_path, 'w') as f:
                json.dump(best_params, f)


# ====================================================================================================================
# Abstract Prediction Model Evaluator with model implementations
# ====================================================================================================================
class RandomForestPredictorModelEvaluator(
    AbstractPredictionModelEvaluator
):
    """For Evaluating Hyper-Parameters of a Prediction Model.
    on the merged tabular (multi-modal) data set. Uses the RandomForestClassifier

    Methods:
        - random_search_with_cross_validation: Performs a random search with cross-validation on the model.
        - get_best_random_search_model: Returns the best estimator from the random search, but only if the
            random_search_with_cross_validation method was called before.
        - get_best_random_search_parameters: Returns the best parameters from the random search, but only if the
            random_search_with_cross_validation method was called before.
    """
    def create_hyperparameter_space_random(self) -> dict[str, list]:
        """Creates the hyperparameter space for the random search.
        Values included are:
        - max_depth
        - max_leaf_nodes
        - max_features
        - min_samples_leaf
        - min_samples_split
        - n_estimators
        - criterion
        For more information check the documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        Returns:
            dict[str, list]: A dictionary with the hyperparameter names as keys
            and the values as list of possible values.
        """
        parameter_dict = {
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_leaf_nodes': [10, 100, 1000],
            'max_features': ['log2', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
        return parameter_dict

    def _create_base_model(self) -> RandomForestClassifier:
        """Creates a RandomForestClassifier with the random state set
        and returns it.

        Returns:
            RandomForestClassifier: The base model for the evaluation.

        """
        base_model = RandomForestClassifier(random_state=np.random.RandomState(self._random_state))
        return base_model


class SvcPredictorModelEvaluator(
    AbstractPredictionModelEvaluator
):
    """For Evaluating Hyper-Parameters of a Prediction Model.
    on the merged tabular (multi-modal) data set. Uses the Support Vector Classifier (SVC).

    Methods:
        - random_search_with_cross_validation: Performs a random search with cross-validation on the model.
        - get_best_random_search_model: Returns the best estimator from the random search, but only if the
            random_search_with_cross_validation method was called before.
        - get_best_random_search_parameters: Returns the best parameters from the random search, but only if the
            random_search_with_cross_validation method was called before.
    """
    def create_hyperparameter_space_random(self) -> dict[str, list]:
        """Creates the hyperparameter space for the SVC model.
        Look at documentation for more information
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        Returns:
            dict[str, list]: A dictionary with the hyperparameter names as keys
            and the values as list of possible values.
        """
        parameter_dict = {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'degree': [3, 7, 10],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
            'max_iter': [100, 1000, 10000, -1]
        }
        return parameter_dict

    def _create_base_model(self) -> any:
        """Creates an instance of the SVC model with the random state set
        and returns it.

        Returns:
            any: The SVC base model for evaluation.

        """
        base_model = SVC(
            random_state=np.random.RandomState(self._random_state),
            probability=True
        )
        return base_model


class AdaBoostPredictorModelEvaluator(
    AbstractPredictionModelEvaluator
):
    """For Evaluating Hyper-Parameters of a Prediction Model.
    on the merged tabular (multi-modal) data set. Uses the AdaBoost Classifier.

    Methods:
        - random_search_with_cross_validation: Performs a random search with cross-validation on the model.
        - get_best_random_search_model: Returns the best estimator from the random search, but only if the
            random_search_with_cross_validation method was called before.
        - get_best_random_search_parameters: Returns the best parameters from the random search, but only if the
            random_search_with_cross_validation method was called before.
    """
    def get_best_random_search_parameters(self) -> dict | None:
        """If random search was performed and no grid search
        inbetween, the best parameters will be returned from the
        random search.

        Returns:
            dict | None: None if the random search was not performed,
            else the best parameters as dictionary.
        """
        if not self._random_search_performed:
            print('Random search not performed!')
            return None
        else:
            best_est_params = self.predictor.model.best_estimator_.get_params()
            try:
                _ = best_est_params.pop('random_state')
                _ = best_est_params.pop('estimator')
                _ = best_est_params.pop('estimator__random_state')
            except Exception as ex:
                pass
            return best_est_params


    def create_hyperparameter_space_random(self) -> dict[str, list]:
        """Creates the hyperparameter space for the random search.
        For more information check the documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
        Returns:
            dict[str, list]: A dictionary with the hyperparameter names as keys
            and the values as list of possible values.
        """
        decision_tree_estimator = DecisionTreeClassifier(
            random_state=np.random.RandomState(42),
            max_depth=3,
            criterion='gini'
        )
        decision_tree_estimator_2 = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_state),
            max_depth=3,
            criterion='entropy'
        )
        decision_tree_estimator_3 = DecisionTreeClassifier(
            random_state=np.random.RandomState(self._random_state),
            max_depth=3,
            criterion='log_loss'
        )
        log_regressor = LogisticRegression(
            random_state=np.random.RandomState(self._random_state),
            penalty='l1', C=1.0, solver='saga'
        )

        parameter_dict = {
            'estimator': [
                None, decision_tree_estimator, decision_tree_estimator_2,
                decision_tree_estimator_3, log_regressor
            ],
            'n_estimators': [10, 50, 100, 500, 1000],
            'learning_rate': [10.0, 1.0, 0.1, 0.01, 0.001],
            'algorithm': ['SAMME']
        }
        return parameter_dict


    def _create_base_model(self) -> AdaBoostClassifier:
        """Creates a RandomForestClassifier with the random state set
        and returns it.

        Returns:
            RandomForestClassifier: The base model for the evaluation.

        """
        base_model = AdaBoostClassifier(
            random_state=np.random.RandomState(self._random_state)
        )
        return base_model


class LogisticRegressorPredictorModelEvaluator(
    AbstractPredictionModelEvaluator
):
    """For Evaluating Hyper-Parameters of a Prediction Model.
    on the merged tabular (multi-modal) data set. Uses the Logistic Regression model.

    Methods:
        - random_search_with_cross_validation: Performs a random search with cross-validation on the model.
        - get_best_random_search_model: Returns the best estimator from the random search, but only if the
            random_search_with_cross_validation method was called before.
        - get_best_random_search_parameters: Returns the best parameters from the random search, but only if the
            random_search_with_cross_validation method was called before.
    """
    def create_hyperparameter_space_random(self) -> dict[str, list]:
        """Creates the hyperparameter space for the random search.
        For more information check the documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        Returns:
            dict[str, list]: A dictionary with the hyperparameter names as keys
            and the values as list of possible values.
        """
        parameter_dict = {
            'penalty': ['l2', 'l1', 'elasticnet'],
            'C': [0.1, 1, 10, 100, 1000],
            'l1_ratio': [0.1, 0.5, 0.9],
            'solver': ['saga']
        }
        return parameter_dict


    def _create_base_model(self) -> LogisticRegression:
        """Creates a RandomForestClassifier with the random state set
        and returns it.

        Returns:
            RandomForestClassifier: The base model for the evaluation.

        """
        base_model = LogisticRegression(
            random_state=np.random.RandomState(self._random_state)
        )
        return base_model


# ====================================================================================================================
# Model Comparer
# ====================================================================================================================
class TabularPredictorComparer:
    def __init__(
            self,
            predictor_list: list[AbstractHancockPredictor],
            predictor_names: list[str],
            predictor_short_names: list[str],
            save_flag: bool = False,
            plot_flag: bool = False, random_state: int = 42,
            plot_name: str = 'comparison_label_to_predict'
    ):
        """Creates a comparer for the different optimal models for prediction tasks.
        To change the predictors that should be used, change the _predictors list together with the
        _predictor_names and _predictor_short_names list.

        Args:
            predictor_list (list[AbstractHancockPredictor]): A list of the predictors that should be compared.
            predictor_names (list[str]): The full names of the predictors.
            predictor_short_names (list[str]): The short names of the predictors.
            save_flag (bool, optional): If the results should be saved. Defaults to False.
            plot_flag (bool, optional): If the results should be plotted. Defaults to False
            random_state (int, optional): The random state for the models. Defaults to 42.
            plot_name (str, optional): The name of the plot that should be saved. Defaults to 'comparison_label_to_predict'.
        """
        self._predictors = predictor_list
        self._predictor_names = predictor_names
        self._predictor_short_names = predictor_short_names
        self._save_flag = save_flag
        self._plot_flag = plot_flag
        self._random_state = random_state
        self._plot_name = plot_name
        np.random.seed(random_state)
        self.random_data = np.random.random(size=(1000, 104)) * 3.0
        self._plotter = PredictionPlotter(
            save_flag=save_flag,
            plot_flag=plot_flag,
            save_dir=self._predictors[0].args.results_dir
        )

        base_data = self._predictors[0].prepare_data_for_model(
            df_train_fold=self._predictors[0].df_train,
            df_other_fold=self._predictors[0].df_test
        )[0]
        self._x_train = base_data[0]
        self._y_train = base_data[1]
        self._x_test = base_data[2]
        self._y_test = base_data[3]

    def compare_models(self) -> tuple[list, list]:
        """Function to compare the different models for the adjuvant therapy predictions.

        Returns:
            tuple[dict, dict]: The first dictionary contains the training metrics and
            the second dictionary the test metrics.
        """
        for predictor, name in zip(self._predictors, self._predictor_names):
            _ = predictor.train(
                plot_name=self._plot_name
            )

        average_adjuvant_treatment_predictions = self._random_data_evaluation()
        training_data_performance, metric_names_train = self._data_evaluation(
            labels=self._y_train, data=self._x_train, data_type='Train'
        )
        test_data_performance, metric_names_test = self._data_evaluation(
            labels=self._y_test, data=self._x_test, data_type='Test'
        )

        train_metrics = self._create_content_for_tabular_visualization(
            average_random_data_performance=average_adjuvant_treatment_predictions,
            data_performance=training_data_performance,
            data_performance_metrics=metric_names_train,
            data_type='Train'
        )

        test_metrics = self._create_content_for_tabular_visualization(
            average_random_data_performance=average_adjuvant_treatment_predictions,
            data_performance=test_data_performance,
            data_performance_metrics=metric_names_test,
            data_type='Test'
        )

        self._plotter.predictor_comparison_table(
            metrics_data=train_metrics[1], row_labels=train_metrics[2],
            metrics_labels=train_metrics[0], fig_size=(30, 6),
            plot_name=f'{self._plot_name}_train_only'
        )

        self._plotter.predictor_comparison_table(
            metrics_data=test_metrics[1], row_labels=test_metrics[2],
            metrics_labels=test_metrics[0], fig_size=(30, 6),
            plot_name=f'{self._plot_name}_test_only'
        )
        return train_metrics, test_metrics

    def _random_data_evaluation(self) -> list[np.ndarray]:
        """
        Predicts the random data with all the predictors and returns the
        mean of the predictions.
        """
        average_predictions = []
        for predictor in self._predictors:
            predictions = predictor.predict(self.random_data)
            average_predictions.append(np.mean(predictions))
        return average_predictions

    def _data_evaluation(
            self, labels: np.array, data: np.ndarray | pd.DataFrame, data_type: str = 'Train'
    ) -> tuple[list[list[int]], list]:
        """
        Compares the predictions from all predictors with the given labels for the data.

        Args:
            labels (np.array): The ground truth labels for the given data. Should be only 0's and
            1's

            data (np.ndarray | pd.DataFrame): The data for the given labels. Should be the same
            dimensionality as the data used to train the predictors.

            data_type (str, optional): The type of the data only used for the metric_names.
            Defaults to Train.

        Returns:
            tuple[list[list[int]], list[str]]: For each predictor a list will be returned with the following metrics
            int that order [num_pos_pred, num_neg_pred, num_tp, num_fp, num_tn, num_fn,
            num_pos_as_predictor1, num_neg_as_predictor1, num_pos_as_predictor2, num_neg_as_predictor2,
            ...]

            num_pos_pred (int): Number of positive predictions for the given data

            num_neg_pred (int): Number of negative predictions for the given data

            num_tp / num_fp / num_tn / num_fn (int): Number of true positive, false positive,
            true negative and false negative.

            num_pos_as_predictorX (int): Number of positive predictions that were also done by
            predictor number X.

            num_neg_as_predictorX (int): Number of negative predictions that were also done by
            predictor number X.

            The second entry in the tuple is a list of the metric names.
        """
        y_predictions = []
        evaluation_results = []
        rounding_acc = 2

        for predictor in self._predictors:
            y_pred_raw = predictor.predict(data)
            y_pred = (y_pred_raw > 0.5).astype(int)
            y_predictions.append(y_pred)
            num_pos = len(np.where(y_pred == 1)[0])
            num_neg = len(np.where(y_pred == 0)[0])
            num_tp = len(np.where(np.logical_and(
                labels == y_pred, labels == 1)
            )[0])
            num_fp = len(np.where(np.logical_and(
                labels != y_pred, labels == 0)
            )[0])
            num_tn = len(np.where(np.logical_and(
                labels == y_pred, labels == 0)
            )[0])
            num_fn = len(np.where(np.logical_and(
                labels != y_pred, labels == 1)
            )[0])
            tpr = self._calculate_true_positive_rate(tp=num_tp, fp=num_fp)
            fpr = self._calculate_false_positive_rate(fp=num_fp, tn=num_tn)
            accuracy = self._calculate_accuracy(tp=num_tp, tn=num_tn, fp=num_fp, fn=num_fn)
            precision = self._calculate_precision(tp=num_tp, fp=num_fp)
            recall = self._calculate_recall(tp=num_tp, fn=num_fn)
            specificity = self._calculate_specificity(tn=num_tn, fp=num_fp)
            f1_score = self._calculate_f1_score(tp=num_tp, fp=num_fp, fn=num_fn)
            dice_score = self._calculate_dice_score(tp=num_tp, fp=num_fp, fn=num_fn)
            auc_score = roc_auc_score(y_true=labels, y_score=y_pred_raw)


            evaluation_results.append([
                num_pos, num_neg, num_tp, num_fp, num_tn, num_fn,
                np.round(tpr, rounding_acc), np.round(fpr, rounding_acc),
                np.round(accuracy, rounding_acc), np.round(precision, rounding_acc),
                np.round(recall, rounding_acc), np.round(specificity, rounding_acc),
                np.round(f1_score, rounding_acc), np.round(dice_score, rounding_acc),
                np.round(auc_score, rounding_acc)
            ])

        for index, y_pred in enumerate(y_predictions):
            for y_other_pred in y_predictions:
                num_also_pos = len(np.where(np.logical_and(
                    y_other_pred == y_pred, y_pred == 1)
                )[0])
                num_also_neg = len(np.where(np.logical_and(
                    y_other_pred == y_pred, y_pred == 0)
                )[0])
                evaluation_results[index].append(num_also_pos)
                evaluation_results[index].append(num_also_neg)

        metrics_names = ['# pos. pred.', '# neg. pred.',
                         f'TP {data_type}', f'FP {data_type}',
                         f'TN {data_type}', f'FN {data_type}',
                         f'TPR {data_type}', f'FPR {data_type}',
                         f'ACC {data_type}', f'PPV {data_type}',
                         f'REC {data_type}', f'SPEC {data_type}',
                         f'F1 {data_type}', f'DICE {data_type}',
                         f'AUC {data_type}']
        for short_name in self._predictor_short_names:
            metrics_names += [f'# pos. as {short_name}', f'# neg. as {short_name}']

        return evaluation_results, metrics_names

    def _create_content_for_tabular_visualization(
            self, average_random_data_performance: list[np.ndarray],
            data_performance: list[list[int]],
            data_performance_metrics: list[str], data_type: str = 'Train'
    ) -> list:
        """Creates the content that is necessary to display the evaluation results in
        a table.

        Args:
            average_random_data_performance (list[np.ndarray]): For each predictor in self._predictors
            the mean prediction on the random data.

            data_performance (list[list[int]]): For each predictor in self._predictors the evaluation
            results on the given data. Type of data should be indicated by data_type

            data_performance_metrics (list[str]): The names of the metrics that are used for the evaluation.
            Should have the same length as the data_performance for a single predictor.

            data_type (str, optional): The name of the data used to create the data_performance.
            Defaults to 'Train'.
        """
        metrics_names = [f'Average prediction random data'] + data_performance_metrics
        metrics_data, row_labels = self._create_metric_data_and_row_labels_for_tabular_visualization(
            data_performance=data_performance, average_random_data_performance=average_random_data_performance
        )

        return [metrics_names, metrics_data, row_labels]

    def _create_metric_data_and_row_labels_for_tabular_visualization(
            self, data_performance: list[list[int]], average_random_data_performance: list[np.ndarray]
    ) -> tuple[list, list]:
        """Creates a single list for the performance and also the row labels based on the
        _predictor_names and _predictor_short_names.

        Args:
            data_performance (list[list[int]]): The performance of the predictors on the given data.

            average_random_data_performance (list[np.ndarray]): The mean prediction for class 1 on the
            random data.

        Returns:
            tuple[list, list]: The first list contains the performance data merged and the second list the
            row_labels
        """
        metrics_data = []
        row_labels = []
        for average, data_performance, full_name, short_name in \
                zip(average_random_data_performance, data_performance,
                    self._predictor_names, self._predictor_short_names):
            metrics_data.append([np.round(average, 2)] + data_performance)
            row_labels.append(f'{full_name} ({short_name})')
        return metrics_data, row_labels

    # ================================================================================================================
    ## Evaluation Metrics
    # ================================================================================================================
    @staticmethod
    def _calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
        """Calculates the accuracy based on the true positive, true negative,
        false positive and false negative.

        Args:
            tp (int): # True positive
            tn (int): # True negative
            fp (int): # False positive
            fn (int): # False negative

        Returns:
            float: The accuracy of the model **not** in percentage.
        """
        if tp + tn + fp + fn == 0:
            return np.nan
        return float(tp + tn) / float(tp + tn + fp + fn)

    @staticmethod
    def _calculate_precision(tp: int, fp: int) -> float:
        """Calculates the precision based on the true positive and false positive.

        Args:
            tp (int): # True positive
            fp (int): # False positive
        """
        if tp + fp == 0:
            return np.nan
        return float(tp) / float(tp + fp)

    @staticmethod
    def _calculate_recall(tp: int, fn: int) -> float:
        """Calculates the recall based on the true positive and false negative.

        Args:
            tp (int): # True positive
            fn (int): # False negative
        """
        if tp + fn == 0:
            return np.nan
        return float(tp) / float(tp + fn)

    @staticmethod
    def _calculate_specificity(tn: int, fp: int) -> float:
        """Calculates the specificity based on the true negative and false positive.

        Args:
            tn (int): # True negative
            fp (int): # False positive
        """
        if tn + fp == 0:
            return np.nan
        return float(tn) / float(tn + fp)

    def _calculate_f1_score(self, tp: int, fp: int, fn: int) -> float:
        """Calculates the F1 score based on the true positive, true negative,
        false positive and false negative.

        Args:
            tp (int): # True positive
            fp (int): # False positive
            fn (int): # False negative
        """
        pre = self._calculate_precision(tp=tp, fp=fp)
        rec = self._calculate_recall(tp=tp, fn=fn)
        if np.isnan(pre) or np.isnan(rec) or pre + rec == 0:
            return np.nan
        return 2 * ((pre * rec) / (pre + rec))

    @staticmethod
    def _calculate_dice_score(tp: int, fp: int, fn: int) -> float:
        """Calculates the Dice score based on the true positive, false positive
        and false negative.

        Args:
            tp (int): # True positive
            fp (int): # False positive
            fn (int): # False negative
        """
        if tp + fp + fn == 0:
            return np.nan
        return 2 * float(tp) / (2.0 * tp + fp + fn)

    @staticmethod
    def _calculate_false_positive_rate(tn: int, fp: int) -> float:
        """Calculates the false positive rate based on the false positive and true negative.

        Args:
            fp (int): # False positive
            tn (int): # True negative
        """
        if fp + tn == 0:
            return np.nan
        return float(fp) / float(fp + tn)

    @staticmethod
    def _calculate_true_positive_rate(tp: int, fp: int):
        """Calculates the true positive rate based on the true positive and false positive.

        Args:
            tp (int): # True positive
            fp (int): # False positive
        """
        if tp + fp == 0:
            return np.nan
        return float(tp) / float(tp + fp)


# ====================================================================================================================
# Metrics to Latex Table
# ====================================================================================================================
class ModelComparisonMetricsToLatexTable:
    """
    Creates from the metrics of the model comparison a latex table that can be used to display the metrics in a table.
    """
    def __init__(
            self,
            metrics: list[tuple[list, list]], data_splits: list[str],
            prediction_target: str = 'Survival Status'
    ):
        self.test_metrics = [metric[1] for metric in metrics]
        self.data_splits = data_splits
        self.prediction_target = prediction_target
        self.relevant_metrics = ['TPR', 'FPR', 'ACC', 'PPV', 'REC', 'SPEC', 'F1', 'DICE', 'AUC']
        self._check_handed_over_parameters()

    def create_latex_table_string(self) -> str:
        """
        Creates a latex table string that can be used to display the metrics in a table.

        Returns:
            str: The latex table string.
        """
        table_content_list = []
        used_metric_names = []
        for test_metric, data_split in zip(self.test_metrics, self.data_splits):
            table_content, used_metric_names = self._parse_metric_to_table_string(
                metric=test_metric, data_split=data_split
            )
            table_content_list.append(table_content)

        header_row = self._create_latex_table_header_row(used_metric_names=used_metric_names)
        table_content = '\\hline \n' + '\n\\hline \n'.join(table_content_list)
        table_content = header_row + table_content
        return table_content

    @staticmethod
    def _create_latex_table_header_row(used_metric_names: list[str]) -> str:
        """
        Creates the header row for the latex table from the used metric names. The header row will look like this:
            Data Split & Model & {used_metric_names[0]} & {used_metric_names[1]} & ...
            \hline

        Args:
            used_metric_names (list[str]): The names of the metrics that should be used for the header row.
        """
        header_row = 'Data Split & Model & ' + ' & '.join(used_metric_names) + ' \\\\ \n\\hline \n'
        return header_row

    def _check_handed_over_parameters(self) -> None:
        """
        Checks if the parameters the class receives are correct.

        Raises:
            ValueError: If the number of data splits is not equal to the number of test metrics.
            ValueError: If the number of predictor names is not equal to the number of predictors.
        """
        if len(self.data_splits) != len(self.test_metrics):
            raise ValueError('The number of data splits is not equal to the number of test metrics!')

    def _parse_metric_to_table_string(self, metric: list[list], data_split: str) -> tuple[str, list]:
        """
        Parses the metric to a table string that can be used in a latex table. The table string will look something
        like this:
        {data_split} & {predictor_names[0] & {metric_values[0][0]} & {metric_values[0][1]} & ... \\
                     & {predictor_names[1] & {metric_values[1][0]} & {metric_values[1][1]} & ... \\

        Args:
            metric (list[list]): The metric that should be parsed to a table string.
            data_split (str): The name of the data split that was used to acquire the metrics

        Returns:
            tuple[str, list]: The first entry is the table string and the second entry is a list of the used metric names.
        """
        predictor_names = metric[2]
        relevant_metrics = self.relevant_metrics
        metric_description = metric[0]
        metric_values = metric[1]
        data_split_string = f'{data_split} '
        used_metric_names = []
        predictor_metrics = []
        for predictor_index, predictor in enumerate(predictor_names):
            predictor_metrics = []
            predictor_string = f'& {predictor} '
            for metric_index, metric_name in enumerate(metric_description):
                if metric_name.split(' ')[0] in relevant_metrics:
                    predictor_metrics.append(metric_name.split(' ')[0])
                    predictor_string += f'& {"{0:0.2f}".format(metric_values[predictor_index][metric_index])} '
            predictor_string += '\\\\\n'
            data_split_string += predictor_string
            used_metric_names.append(predictor_metrics)
        return data_split_string, predictor_metrics




