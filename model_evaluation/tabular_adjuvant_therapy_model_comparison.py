import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning import TabularAdjuvantTreatmentPredictor, PredictionPlotter


class OptimalRandomForestTabularAdjuvantTreatmentPredictor(
    TabularAdjuvantTreatmentPredictor
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
    TabularAdjuvantTreatmentPredictor
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
    TabularAdjuvantTreatmentPredictor
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
        for predictor, name in zip(self._predictors, self._predictor_names):
            _ = predictor.train(
                plot_name='comparison_adjuvant_treatment_multimodal'
            )

        average_adjuvant_treatment_predictions = self._random_data_evaluation()
        training_data_performance = self._data_evaluation(
            labels=self._y_train, data=self._x_train
        )

        self._create_comparison(
            average_random_data=average_adjuvant_treatment_predictions,
            training_data_performance=training_data_performance
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
            self, labels: np.array, data: np.ndarray | pd.DataFrame
    ) -> list[list[int]]:
        """
        Compares the predictions from all predictors with the given labels for the data.

        Args:
            labels (np.array): The ground truth labels for the given data. Should be only 0's and
            1's
            data (np.ndarray | pd.DataFrame): The data for the given labels. Should be the same
            dimensionality as the data used to train the predictors.

        Returns:
            list[list[int]]: For each predictor a list will be returned with the following metrics
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

        return evaluation_results

    def _create_comparison(
            self, average_random_data: list[np.ndarray],
            training_data_performance: list[list[int]],
    ) -> None:
        if len(average_random_data) != len(training_data_performance) != len(self._predictors):
            raise ValueError(
                "The length of the given data does not match the number of predictors."
            )
        metrics_data = []
        row_labels = []

        data_performance_metrics = ['# pos. pred', '# neg. pred', 'TP Train', 'FP Train', 'TN Train', 'FN Train']
        for short_name in self._predictor_short_names:
            data_performance_metrics += [f'# pos. as {short_name}', f'# neg. as {short_name}']
        average_sign = "\u03BC"
        metrics = [f'Average prediction random data'] + data_performance_metrics

        for average, data_performance, full_name, short_name in \
                zip(average_random_data, training_data_performance, self._predictor_names, self._predictor_short_names):
            metrics_data.append([np.round(average, 2)] + data_performance)
            row_labels.append(f'{full_name} ({short_name})')

        self._plotter.predictor_comparison_table(
            metrics_data=metrics_data, row_labels=row_labels,
            metrics=metrics, fig_size=(20, 6),
            plot_name='comparison_adjuvant_treatment_multimodal_train_only'
        )


if __name__ == '__main__':
    comparer = TabularAdjuvantTherapyComparer(
        save_flag=True, plot_flag=True, random_state=42
    )
    comparer.compare_models()

