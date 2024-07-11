import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import json

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning import TabularAdjuvantTreatmentPredictor


class TabularAdjuvantTreatmentModelEvaluator:
    def __init__(
            self, plot_flag: bool = False, save_flag: bool = False, random_state: int = 42
    ):
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
            self.save_random_search_results(file_name=plot_name)
        return {'train_returns': train_returns}

    def get_best_random_search_model(self) -> any:
        if not self._random_search_performed:
            print('Random search not performed!')
            return None
        return self.predictor.model.best_estimator_

    def get_best_random_search_parameters(self) -> dict | None:
        if not self._random_search_performed:
            print('Random search not performed!')
            return None
        else:
            return self.predictor.model.best_params_

    def save_random_search_results(
            self, file_name: str = 'random_search_adjuvant_therapy'
    ) -> None:
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
        return {'Name': np.NAN}

    @staticmethod
    def create_hyper_parameter_space_grid() -> dict[str, any]:
        return {'Name': np.NAN}

    def _create_base_model(self) -> any:
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

