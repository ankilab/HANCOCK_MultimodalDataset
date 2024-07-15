import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning import TabularAdjuvantTherapyPredictor


class OptimalRandomForestTabularAdjuvantTherapyPredictor(
    TabularAdjuvantTherapyPredictor
):
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


class OptimalSVCTabularAdjuvantTherapyPredictor(
    TabularAdjuvantTherapyPredictor
):
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
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class OptimalAdaBoostTabularAdjuvantTherapyPredictor(
    TabularAdjuvantTherapyPredictor
):
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


class TabularAdjuvant