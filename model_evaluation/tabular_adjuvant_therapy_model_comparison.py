import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning import TabularMergedAdjuvantTreatmentPredictor, PredictionPlotter


class OptimalRandomForestTabularAdjuvantTreatmentPredictor(
    TabularMergedAdjuvantTreatmentPredictor
):
    """
    Optimal Random Forest model for predicting adjuvant therapy on the
    merged tabular data. Uses the model that is optimal for the given trainings
    data with 10-cross validation and grid search.
    """
    def __init__(
            self, save_flag: bool = False, plot_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            save_flag=save_flag,
            plot_flag=plot_flag,
            random_state=random_state
        )

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
    data with 10-cross validation and grid search.
    """
    def __init__(
            self, save_flag: bool = False, plot_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            save_flag=save_flag,
            plot_flag=plot_flag,
            random_state=random_state
        )

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
    data with 10-cross validation and grid search.
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


class TabularAdjuvantTherapyComparer:
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
        self._predictors = [
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
        ]
        self._predictor_names = ['Random Forest', 'Ada Boost', 'Support Vector Classifier']
        self._predictor_short_names = ['RF', 'Ada', 'SVC']
        self._save_flag = save_flag
        self._plot_flag = plot_flag
        self._random_state = random_state
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

    def compare_models(self):
        """Function to compare the different models for the adjuvant therapy predictions.
        """
        for predictor, name in zip(self._predictors, self._predictor_names):
            _ = predictor.train(
                plot_name='comparison_adjuvant_treatment_multimodal'
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
            metrics_labels=train_metrics[0], fig_size=(20, 6),
            plot_name='comparison_adjuvant_treatment_multimodal_train_only'
        )

        self._plotter.predictor_comparison_table(
            metrics_data=test_metrics[1], row_labels=test_metrics[2],
            metrics_labels=test_metrics[0], fig_size=(20, 6),
            plot_name='comparison_adjuvant_treatment_multimodal_test_only'
        )

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

        for predictor in self._predictors:
            y_pred = predictor.predict(data)
            y_pred = (y_pred > 0.5).astype(int)
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
            evaluation_results.append([
                num_pos, num_neg, num_tp, num_fp, num_tn, num_fn
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
                         f'TN {data_type}', f'FN {data_type}']
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


if __name__ == '__main__':
    comparer = TabularAdjuvantTherapyComparer(
        save_flag=True, plot_flag=True, random_state=42
    )
    comparer.compare_models()

