import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import json

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning import TabularAdjuvantTreatmentPredictor


class TabularAdjuvantTreatmentModelEvaluator:
    """For Evaluating Hyper-Parameters of Adjuvant Treatment Models
    on the merged tabular (multi-modal) data set.
    """
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ) -> None:
        """Creates an instance of the TabularAdjuvantTreatmentModelEvaluator. Should not be used
        directly because this is intended to be an abstract class.

        Args:
            plot_flag (bool, optional): Flag to plot the results. Defaults to False.
            save_flag (bool, optional): Flag to save the results. Defaults to False
            random_state (int, optional): Random state seed for reproducibility. Will be passed to
            all functions classes that contain random events. Defaults to 42.
        """
        self.predictor = TabularAdjuvantTreatmentPredictor(
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )
        self.hyper_parameter_space_random = self.create_hyper_parameter_space_random()
        self.hyper_parameter_space_grid = self.create_hyper_parameter_space_grid()
        self._random_state = random_state
        self._random_search_performed = False
        self._grid_search_performed = False
        self._save_flag = save_flag
        self._plot_flag = plot_flag

    # Random Search
    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100, scoring: str = 'f1',
            plot_name: str = 'random_search_adjuvant_therapy'
    ) -> dict[str, any]:
        """Performs a random search with cross-validation on the model
        (call of _create_base_model()) using the hyper-parameter space
        defined (call of create_hyper_parameter_space_random()).

        Args:
            n_splits (int, optional): Number of splits for cross-validation.
            Defaults to 10.

            n_iter (int, optional): Number of iterations for the random search.
            Will choose n_iter hyperparameter combinations. Defaults to 100.

            scoring (str, optional): Scoring metric for the selection of the best model.
            For more metrics check
            https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.
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

    @staticmethod
    def create_hyper_parameter_space_random() -> dict[str, any]:
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


class RandomForestTabularAdjuvantTreatmentModelEvaluator(
    TabularAdjuvantTreatmentModelEvaluator
):
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
        super().__init__(
            plot_flag=plot_flag, save_flag=save_flag, random_state=random_state
        )

    def random_search_with_cross_validation(
            self, n_splits: int = 10, n_iter: int = 100,
            scoring: str = 'f1',
            plot_name: str = 'random_search_random_forest_adjuvant_therapy_multimodal'
    ) -> dict[str, any]:
        return super().random_search_with_cross_validation(
            n_splits=n_splits, n_iter=n_iter, scoring=scoring,
            plot_name=plot_name
        )

    @staticmethod
    def create_hyper_parameter_space_random() -> dict[str, any]:
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
        base_model = RandomForestClassifier(random_state=np.random.RandomState(self._random_state))
        return base_model


if __name__ == "__main__":
    main_plot_flag = True
    main_save_flag = True
    test_n_splits = 10
    test_n_iter = 100
    random_forest_evaluator = RandomForestTabularAdjuvantTreatmentModelEvaluator(
        plot_flag=main_plot_flag, save_flag=main_save_flag
    )
    _ = random_forest_evaluator.random_search_with_cross_validation(
        n_splits=test_n_splits, n_iter=test_n_iter
    )

