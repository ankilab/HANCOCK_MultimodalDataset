# ===================================================================================================================
# Imports
# ====================================================================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))
from multimodal_machine_learning.predictor.abstract import (
    AbstractHancockPredictor
)
from defaults import DefaultNames


# ===================================================================================================================
# Abstract Survival Status Predictor without model implementation and data split preparation
# ====================================================================================================================
class AbstractTabularMergedSurvivalStatusPredictor(AbstractHancockPredictor):
    def __init__(
            self, save_flag: bool = False, plot_flag: bool = False,
            random_state: int =42
    ):
        """
        Initializes the AbstractSurvivalStatusPredictor. This class is an abstract class that is used to predict
        the survival status of a patient using the Hancock dataset.

        Args:
            save_flag (bool, optional): If this is set to True the generated
            plots will be saved. Defaults to False.

            plot_flag (bool, optional): If this is set to True the generated
            plots will be shown to the user. Defaults to False.

            random_state (int, optional): To make sure that the outputs are
            reproducible we initialize all processes that involve randomness
            with the random_state. Defaults to 42.

        Abstract Methods:
            _prepare_data_split: The data split getting process. Should set the self._data_split property.
            _create_new_model: The method that should create a new model that is used for the prediction tasks.
        """
        self._df_split = None
        super().__init__(save_flag=save_flag, plot_flag=plot_flag,
                         random_state=random_state,
                         predictor_type='survival_status_prediction')

    def _prepare_data(self) -> None:
        """
        Prepares the self._data and the self._targets for the survival status prediction.
        Uses the tabular merged data and the targets.csv.
        """
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tabular_merged_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True
        )
        self._data = data_reader.return_data()

        targets_dir = self.args.features_dir / DefaultNames().targets
        if not targets_dir.exists():
            raise FileNotFoundError(f"Could not find the targets directory: {targets_dir}")

        data_reader_targets = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.targets,
            data_dir=targets_dir, data_dir_flag=True
        )
        self._targets = data_reader_targets.return_data()

    def _get_df_train(self) -> pd.DataFrame:
        """
        Returns the training data for the survival status prediction.
        """
        df_split = self._return_df_split()
        df_train = df_split[df_split.dataset == "training"][["patient_id", 'survival_status']].copy()
        df_train.columns = ["patient_id", "target"]
        df_train = df_train.merge(self._data, on='patient_id', how='inner')
        return df_train

    def _get_df_test(self) -> pd.DataFrame:
        df_split = self._return_df_split()
        df_test = df_split[df_split.dataset == "test"][["patient_id", 'survival_status']].copy()
        df_test.columns = ["patient_id", "target"]
        df_test = df_test.merge(self._data, on="patient_id", how="inner")
        return df_test

    # =================================================================================================================
    # Helper Methods
    # =================================================================================================================
    def _return_df_split(self) -> pd.DataFrame:
        """
        Returns the data split for the survival status prediction. This method combines the self._targets and
        self._data_split data frames. The df_split data frame is used in the _get_df_train and _get_df_test methods.

        Returns:
            pd.DataFrame: The data split for the survival status prediction with the targets.
        """
        if self._df_split is not None:
            return self._df_split
        self._df_split = self._data_split.merge(self._targets, on='patient_id', how='inner')
        self._df_split = self._df_split[~(self._df_split.survival_status_with_cause == "deceased not tumor specific")]
        self._df_split['survival_status'] = self._df_split['survival_status'].replace({"living": 0, "deceased": 1})
        return self._df_split


# ===================================================================================================================
# Abstract Survival Status Predictor without model implementation
# ====================================================================================================================
class AbstractTabularMergedSurvivalStatusPredictorInDistribution(AbstractTabularMergedSurvivalStatusPredictor):
    def _prepare_data_split(self) -> None:
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_in,
            data_dir=self.args.data_split_dir / self._default_names.data_split_in,
            data_dir_flag=True
        )
        self._data_split = data_split_reader.return_data()


class AbstractTabularMergedSurvivalStatusPredictorOutDistribution(AbstractTabularMergedSurvivalStatusPredictor):
    def _prepare_data_split(self) -> None:
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_out,
            data_dir=self.args.data_split_dir / self._default_names.data_split_out,
            data_dir_flag=True
        )
        self._data_split = data_split_reader.return_data()


class AbstractTabularMergedSurvivalStatusPredictorOropharynx(AbstractTabularMergedSurvivalStatusPredictor):
    def _prepare_data_split(self) -> None:
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_in,
            data_dir=self.args.data_split_dir / self._default_names.data_split_oropharynx,
            data_dir_flag=True
        )
        self._data_split = data_split_reader.return_data()


# ===================================================================================================================
# Survival Status Predictor
# ====================================================================================================================
class TabularMergedSurvivalStatusPredictorInDistribution(AbstractTabularMergedSurvivalStatusPredictorInDistribution):
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=500, random_state=np.random.RandomState(self._random_number)
        )
        return model


class TabularMergedSurvivalStatusPredictorOutDistribution(AbstractTabularMergedSurvivalStatusPredictorOutDistribution):
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=500, random_state=np.random.RandomState(self._random_number)
        )
        return model


class TabularMergedSurvivalStatusPredictorOropharynx(AbstractTabularMergedSurvivalStatusPredictorOropharynx):
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=500, random_state=np.random.RandomState(self._random_number)
        )
        return model

