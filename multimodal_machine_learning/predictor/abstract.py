# ===================================================================================================================
# Imports
# ====================================================================================================================
import numpy as np
import pandas as pd
import warnings
import shap
from matplotlib import rcParams
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from lime import lime_tabular
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

from data_reader import DataFrameReaderFactory
from argument_parser import HancockArgumentParser
from defaults import DefaultNames
from multimodal_machine_learning.custom_preprocessor import HancockTabularPreprocessor
from multimodal_machine_learning.predictor.plotter import PredictionPlotter


# ===================================================================================================================
# Base Predictor
# ====================================================================================================================
class AbstractHancockPredictor:
    """
    Abstract Class for performing prediction tasks with the Hancock Data. Provides the basic implementation for
    cross-validation, training and prediction tasks. The data will be preprocessed with the HancockTabularPreprocessor.

    Methods:
        cross-validation: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the label for the given data. The data should be preprocessed with the helper method
        prepare_data_for_model.

    Abstract Methods:
        _prepare_data: The data getting process. Should set the self._data property.
        _prepare_data_split: The data split getting process. Should set the self._data_split property.
        _get_df_train: The method that should return the training data.
        _get_df_test: The method that should return the test data.
        _create_new_model: The method that should create a new model that is used for the prediction tasks.
    """
    @property
    def df_train(self) -> pd.DataFrame:
        """Data frame for the trainings data. Will be eventually split into
        training and validation data, depending on the implementation.
        Uses the _get_df_train method to get the data the first time it is called.

        Returns:
            pd.DataFrame: The trainings data as a pandas data frame.
        """
        if self._df_train is None:
            self._df_train = self._get_df_train()
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        """Data frame for only the test data. Do not use it for training,
        validation etc. Uses the _get_df_test method to get the data the first
        time it is called.

        Returns:
            pd.DataFrame: The test data as a pandas data frame.
        """
        if self._df_test is None:
            self._df_test = self._get_df_test()
        return self._df_test.copy()

    @property
    def model(self):
        """The model that is used to make the predictions. Uses the _get_model
        method to get the model the first time it is called.

        Returns:
            model: The model that is used to make the predictions. After
            the model is trained, it is a copy of the trained model.
        """
        if self._model is None:
            self._model = self._get_model()
        return self._model

    @model.setter
    def model(self, model):
        """Setter for the model property. If the setter is used, the model
        will no longer be initialized by the _create_new_model method but instead
        will use the model that is set here.
        If this process should be discarded and the initial model should be
        used, first set the self.model_setter_flag to False and then call this
        setter with None.

        Args:
            model: The model that should be used to make the predictions.
        """
        if model is not None:
            self.model_setter_flag = True
        elif model is None:
            self._model = model
            if not self.model_setter_flag:
                self._untrained_model = model
                self.model_setter_flag = False
            return

        self._untrained_model = copy.deepcopy(model)
        self._model = model

    def reset_model(self):
        """Resets the model so that it again uses the default
        model. Keep in mind that it also deletes the training process.
        """
        self.model_setter_flag = False
        self.model = None


    def __init__(
            self, save_flag: bool = False, plot_flag: bool = False,
            random_state: int = 42, predictor_type: str = 'None'
    ):
        """
        Initializes the AdjuvantTreatmentPredictor. It can be used to
        perform adjuvant treatment prediction with the merged tabular data.

        Args:
            save_flag (bool, optional): If this is set to True the generated
            plots will be saved. Defaults to False.

            plot_flag (bool, optional): If this is set to True the generated
            plots will be shown to the user. Defaults to False.

            random_state (int, optional): To make sure that the outputs are
            reproducible we initialize all processes that involve randomness
            with the random_state. Defaults to 42.
        """
        rcParams.update({"font.size": 6})
        rcParams["svg.fonttype"] = "none"
        self.argumentParser = HancockArgumentParser(
            file_type=predictor_type)
        self.args = self.argumentParser.parse_args()
        self._save_flag = save_flag
        self._plot_flag = plot_flag
        self._plotter = PredictionPlotter(
            self.args.results_dir, save_flag, plot_flag)
        self._random_number = random_state
        self._data_reader_factory = DataFrameReaderFactory()
        self._default_names = DefaultNames()

        self._df_train = None
        self._df_test = None
        self._model = None
        self._untrained_model = None
        self.model_setter_flag = False

        self._prepare_data()
        self._prepare_data_split()
        self._preprocessor = None
        self._prepare_preprocessor()

    # ================================================================================================================
    # Cross Validate
    # ================================================================================================================
    def cross_validate(
            self, n_splits: int = 10, plot_name: str = 'adjuvant_therapy_multimodal', **kwargs
    ) -> list[list]:
        """Performs cross-validation on the training data (df_train) with
        n_split folds. The cross-validation is done with a StratifiedKFold.

        Args:
            n_splits (int): The number of splits for the cross-validation.
            Defaults to 10.
            plot_name (str): The name of the plot that should be saved to disk.

        Returns:
            list[list]: A list with [tpr_list, auc_list, val_index_list,
            features_per_fold, shap_values]
            tpr_list: List with the true positive rates for each fold.
            auc_list: List with the AUC scores for each fold.
            val_index_list: List with the indices of the validation data for each fold.
            features_per_fold: List with the ColumnTransformer features for each fold.
        """
        tpr_list = []
        auc_list = []
        shap_values = []
        val_index_list = []
        features_per_fold = []

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=np.random.RandomState(self._random_number))
        self._plotter.deactivate()

        for train_idx, val_idx in cv.split(self.df_train, self.df_train['target']):
            val_index_list.append(val_idx)
            df_train_fold = self.df_train.iloc[train_idx]
            df_val_fold = self.df_train.iloc[val_idx]
            self._cross_validation_single_fold(
                df_train_fold, df_val_fold, tpr_list, auc_list, shap_values,
                features_per_fold, plot_name
            )

        self._plotter.reactivate()
        self._plotter.roc_curve(auc_list, tpr_list, plot_name)

        return [tpr_list, auc_list, val_index_list, features_per_fold, shap_values]

    def _cross_validation_single_fold(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame,
            tpr_list: list, auc_list: list, shap_values: list,
            features_per_fold: list, plot_name: str, **kwargs
    ) -> None:
        """Performs the training for a single fold and adds the results to the
        corresponding lists.

        Args:
            df_train_fold (pd.DataFrame): The data that should be used for training.
            df_val_fold (pd.DataFrame): The data that should be used for validation.
            tpr_list (list): List with the true positive rates for each fold.
            auc_list (list): List with the AUC scores for each fold.
            shap_values (list): List with the SHAP values for each fold.
            features_per_fold (list): List with the ColumnTransformer features
            plot_name (str): The name of the plot that should be saved to disk
            for the trainings process.
        """
        _ = self._cross_validation_single_fold_train_and_add_basic_metrics(
            df_train_fold, df_val_fold, plot_name, tpr_list, auc_list, features_per_fold
        )
        shap_values.append(None)


    def _cross_validation_single_fold_train_and_add_basic_metrics(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame, plot_name: str,
            tpr_list: list, auc_list: list, features_per_fold: list
    ) -> list:
        """Performs the training for a single fold and adds the basic results to the
        corresponding lists.

        Args:
            df_train_fold (pd.DataFrame): The data that should be used for training.
            df_val_fold (pd.DataFrame): The data that should be used for validation.
            tpr_list (list): List with the true positive rates for each fold.
            auc_list (list): List with the AUC scores for each fold.
            features_per_fold (list): List with the ColumnTransformer features
            plot_name (str): The name of the plot that should be saved to disk
            for the trainings process.
        """
        train_return = self.train(
            df_train=df_train_fold, df_other=df_val_fold, plot_name=plot_name, model_reset=True
        )
        tpr = train_return[0][1]
        tpr_list.append(tpr)

        roc_score = train_return[0][2]
        auc_list.append(roc_score)

        features = train_return[2]
        features_per_fold.append(features)
        return train_return

    # ================================================================================================================
    # Train
    # ================================================================================================================
    def train(
            self, df_train: pd.DataFrame = None, df_other: pd.DataFrame = None,
            plot_name: str = 'adjuvant_treatment_multimodal', model_reset: bool = True, **kwargs
    ) -> list:
        """This method trains the model on the given data and returns
        performance metrics for the validation data as well as the
        data that was used for training and validation.

        Args:
             df_train (pd.DataFrame, optional): The data that should be used
             to train the model. Should have the columns 'patient_id' and 'target'.
             Defaults to None and then the self.df_train property is used.

             df_other (pd.DataFrame, optional): The data that should be used to validate
             the trainings results. Should have the columns 'patient_id' and 'target'.

             Defaults to None and then the self.df_test property.
             plot_name (str, optional): The name of the plot that should be saved.

             model_reset (bool, optional): If this is set to True the model will be
             reset to None before the training process. Defaults to True.

        Returns:
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
        df_train, df_other = self._setup_training(df_train, df_other, model_reset)

        [[x_train, y_train, x_other, y_other], features] = self.prepare_data_for_model(
            df_train, df_other)
        self.model.fit(x_train, y_train)
        y_pred = self.predict(x_other)

        [fpr, tpr, auc_score] = self._training_calculate_metrics(
            y_other, y_pred)

        if self._plotter.plot_flag or self._plotter.save_flag:
            self._plot_train(df_other, y_pred, plot_name)

        return [[fpr, tpr, auc_score],
                [x_train, y_train, x_other, y_other, y_pred], features]

    def _setup_training(
            self, df_train: pd.DataFrame | None, df_other: pd.DataFrame | None, model_reset: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Checks the df_train and df_other for None values. If this is the case
        instead the df_train and df_test properties of the class are used. Also resets
        the model if model_reset is True.

        Args:
            df_train (pd.DataFrame | None): The data that should be used for training
            the model.

            df_other (pd.DataFrame | None): The data that should be used for validation.

            model_reset (bool): If this is set to True the model will be reset to None.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The training and validation data frames.
        """
        if df_train is None:
            df_train = self.df_train
        if df_other is None:
            df_other = self.df_test
        if model_reset:
            self._model = None
        return df_train, df_other

    @staticmethod
    def _training_calculate_metrics(y_other: np.array, y_pred: np.array) -> list:
        """Calculates the true positive rate, false positive rate and the
        AUC score for the training process.

        Args:
            y_other (np.array): The ground truth labels
            y_pred (np.array): The predicted labels

        Returns:
            list: List with [fpr, tpr, auc_score] in that order
            fpr (np.array): False positive rate
            tpr (np.array): True positive rate
            auc_score (float): The AUC score
        """
        if len(np.shape(y_other)) > 1:
            y_other = y_other[:, 1]

        x_linspace = np.linspace(0, 1, 100)
        tpr = np.empty(100)
        fpr = np.empty(100)
        auc_score = 0.0
        if len(np.unique(y_other)) == 1:
            print('Only one class in validation / test data. Manual setting of true positive rate' +
                  ' and false positive rate and AUC Score')
            if y_other[0] == 1:
                fpr[:] = 0.0
                tpr[:] = 1.0
            elif y_other[0] == 0:
                fpr[:] = 1.0
                tpr[:] = 0.0
            else:
                fpr[:] = 0.0
                tpr[:] = 0.0
        else:
            fpr, tpr, _ = roc_curve(y_other, y_pred)
            tpr = np.interp(x_linspace, fpr, tpr)
            tpr[0] = 0.0
            tpr[-1] = 1.0
            auc_score = roc_auc_score(y_other, y_pred)
        print(
            f'Training Classification Report: \n{classification_report(y_other, y_pred > 0.5)}',
            end='\n\n'
        )
        return [fpr, tpr, auc_score]

    def _plot_train(
            self, df_test: pd.DataFrame, y_pred: np.array, plot_name: str
    ) -> None:
        survival_targets = self._data_reader_factory.make_data_frame_reader(
            self._data_reader_factory.data_reader_types.targets_adjuvant_treatment
        ).return_data()
        survival_targets = survival_targets[survival_targets.patient_id.isin(
            df_test.patient_id.tolist())].reset_index(drop=True)

        y_pred = (y_pred > 0.5).astype(int)
        self._plotter.prediction_plot(y_pred, plot_name)

        if sum(y_pred) > 0:
            self._plotter.follow_up_months_plot(
                survival_targets, y_pred, plot_name)
            self._plotter.recurrence_free_survival_plot(
                survival_targets, y_pred, plot_name)
        else:
            warnings.warn('No positive predictions. No plots for follow-up months and recurrence-free survival.')

    # ================================================================================================================
    # Predict
    # ================================================================================================================
    def predict(self, data: pd.DataFrame | np.ndarray) -> np.array:
        y_pred = self.model.predict_proba(data)[:, 1]
        return y_pred

    # ===============================================================================================================
    # Helper Methods
    # ===============================================================================================================
    def _get_model(self):
        if self.model_setter_flag:
            if self._untrained_model is None:
                print('No model set. Either set a model or change the' +
                      ' model_setter_flag to False.')
            return copy.deepcopy(self._untrained_model)
        else:
            return self._create_new_model()

    def _prepare_preprocessor(self) -> None:
        self._preprocessor = HancockTabularPreprocessor(
            self.df_train.drop(['patient_id', 'target'], axis=1).columns
        )
            # self.df_train.drop(['patient_id', 'target']).columns, min_max_scaler=True)
        self._preprocessor = self._preprocessor.fit(
            pd.concat([self.df_train, self.df_test]
                      ).drop(['patient_id', 'target'], axis=1))
    # ===============================================================================================================
    # Abstract Methods
    # ===============================================================================================================
    def _prepare_data(self) -> None:
        """Should implement the data getting process for all data (training and test) and write the data to
        the self._data property. The data should be a pandas data frame.

        Raises:
            NotImplementedError: If the method is not implemented by the
            child class.
        """
        self._data = None
        raise NotImplementedError("Data preparing not implemented")

    def _prepare_data_split(self) -> None:
        """
        Should prepare the self._data_split property for the _get_df_train and
        _get_df_test methods. The data split should be a pandas data frame with
        a column 'patient_id' and 'target'. 'target' should be selected based on
        the intended prediction task.
        Must be implemented in the subclass.

        Raises:
            NotImplementedError: if the method is not implemented by the
            child class.
        """
        self._data_split = None
        raise NotImplementedError("Data split getting not implemented.")

    def _get_df_train(self) -> pd.DataFrame:
        """
        Creates the training data for the predictor. It is possible to use the
        self._data_split property as well as the self._data property.
        The method is used to set the self._df_train property that is used for
        setting the value of the self.df_train property.
        """
        raise NotImplementedError("Data frame for training not implemented.")

    def _get_df_test(self) -> pd.DataFrame:
        """
        Creates the test data for the predictor. It is possible to use the
        self._data_split property as well as the self._data property.
        The method is used to set the self._df_test property that is used for
        setting the value of the self.df_test property.

        Returns:
            pd.DataFrame: The data frame that is contains the test data.
        """
        raise NotImplementedError("Data frame for testing not implemented.")

    def _create_new_model(self) -> any:
        """
        Should create a new model and return it. The model will be used for
        prediction tasks and will be trained in the default schema.
        Has to be implemented in the subclass. The model should have a predict_proba
        method implemented. Otherwise, the predict method has to be overwritten.

        Returns:
            any: The model that is used for prediction tasks etc.
        """
        raise NotImplementedError("Model creation not implemented.")

    # ===============================================================================================================
    # Useful Helper Methods
    # ===============================================================================================================
    def prepare_data_for_model(
            self, df_train_fold: pd.DataFrame, df_other_fold: pd.DataFrame
    ) -> list:
        """Preprocess the input data and creates a numpy array for the labels.
        Also, over-samples the underrepresented class if necessary.

        Args:
            df_train_fold (pd.DataFrame): The data intended for training. Must have
            a column 'patient_id' and 'target'.
            df_other_fold (pd.DataFrame): The data intended for validating or testing the training process.

        Returns:
            list: List with [[x_train, y_train, x_val, y_val], features] in that order.
        """
        x_train = self._preprocessor.transform(
            df_train_fold.drop(["patient_id", "target"], axis=1))
        x_val = self._preprocessor.transform(
            df_other_fold.drop(["patient_id", "target"], axis=1))
        y_train = df_train_fold["target"].to_numpy()
        y_val = df_other_fold["target"].to_numpy()

        # Handle class imbalance
        smote = SMOTE(random_state=np.random.RandomState(self._random_number))
        x_train, y_train = smote.fit_resample(x_train, y_train)

        features = self._preprocessor.get_feature_names_out()
        return [[x_train, y_train, x_val, y_val], features]


class AbstractHancockPredictorShap(AbstractHancockPredictor):
    """
    Abstract Class for the Prediction using the Hancock Data.

    There are already basic routines implemented for the cross_validation, train
    and predict methods. It will use shapley values to visualize the feature importance. This
    is initialized in the cross_validation method. As the shapley library is not compatible with
    deep neural networks this method has to be overwritten depending on the used model.

    To implement:
        - _prepare_data: Data getting process.
        - _prepare_data_split: Getting the correct data split for the training and validation data.
        - _get_df_train: Use the data split and data to return the training data.
        - _get_df_test: Use the data split and data to return the test data.
        - _create_new_model: Create a new model for the prediction process.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    # ================================================================================================================
    # Cross Validate
    # ================================================================================================================
    def cross_validate(
            self, n_splits: int = 10, plot_name: str = 'adjuvant_therapy_multimodal', **kwargs
    ) -> list[list]:
        """Performs cross-validation on the training data (df_train) with
        n_split folds. The cross-validation is done with a StratifiedKFold.

        Args:
            n_splits (int): The number of splits for the cross-validation.
            Defaults to 10.
            plot_name (str): The name of the plot that should be saved to disk.

        Returns:
            list[list]: A list with [tpr_list, auc_list, val_index_list,
            features_per_fold, shap_values]
            tpr_list: List with the true positive rates for each fold.
            auc_list: List with the AUC scores for each fold.
            val_index_list: List with the indices of the validation data for each fold.
            features_per_fold: List with the ColumnTransformer features for each fold.
            shap_values: List with the SHAP values for each fold.
        """
        [tpr_list, auc_list, val_index_list, features_per_fold, shap_values] = super().\
            cross_validate(
                n_splits=n_splits, plot_name=plot_name
            )

        self._cross_validation_shap(shap_values, features_per_fold, val_index_list,
                                    self.df_train, plot_name)
        return [tpr_list, auc_list, val_index_list, features_per_fold, shap_values]

    def _cross_validation_single_fold(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame,
            tpr_list: list, auc_list: list, shap_values: list,
            features_per_fold: list, plot_name: str, **kwargs
    ) -> None:
        """Performs the training for a single fold and adds the results to the
        corresponding lists.

        Args:
            df_train_fold (pd.DataFrame): The data that should be used for training.
            df_val_fold (pd.DataFrame): The data that should be used for validation.
            tpr_list (list): List with the true positive rates for each fold.
            auc_list (list): List with the AUC scores for each fold.
            shap_values (list): List with the SHAP values for each fold.
            features_per_fold (list): List with the ColumnTransformer features
            plot_name (str): The name of the plot that should be saved to disk
            for the trainings process.
        """
        train_return = self._cross_validation_single_fold_train_and_add_basic_metrics(
            df_train_fold, df_val_fold, plot_name, tpr_list, auc_list, features_per_fold
        )
        x_val = train_return[1][2]
        svs = shap.TreeExplainer(self.model).shap_values(x_val)
        shap_values.append(svs[:, :, 1])  # class = 1 (adjuvant treatment yes)

    def _cross_validation_shap(
            self, shap_values: list, features_per_fold: list,
            val_index_list: list, data: pd.DataFrame, plot_name: str
    ) -> None:
        """Processes the SHAP values for the cross-validation. This includes
        fixing missing features for SHAP values in folds where the feature value
        was not present in the training data. Also, plot the SHAP values and
        saves them to disk, if the save_flag and plot_flag are set to True.

        Args:
            shap_values (list): The SHAP values for each fold. If values are
            missing in folds, they will be added through this function.
            features_per_fold (list): The features that were used for training.
            val_index_list (list): The indices of the validation data for each fold.
            data (pd.DataFrame): The data that was used for training and validation.
            plot_name (str): The name of the plot that should be saved.
        """
        [data_preprocessed, feature_names] = self._cross_validation_fix_shap_values(
            shap_values, features_per_fold, data
        )
        all_val_folds_idx = [
            idx for idx_fold in val_index_list for idx in idx_fold]

        self._plotter.plot_shap_values(
            shap_values, data_preprocessed, feature_names, all_val_folds_idx
        )
        self._plotter.save_shap_values(
            shap_values, data_preprocessed, feature_names, all_val_folds_idx, plot_name
        )

    @staticmethod
    def _cross_validation_fix_shap_values(
            shap_values: list, features_per_fold: list, data: pd.DataFrame
    ) -> list[list]:
        """Fix missing features in SHAP values for each fold, because if not every value
        of a feature is present in each fold of the cross validation the SHAP value for
        this feature will be missing. This function adds zeros for missing features.

        Args:
            shap_values (list): The SHAP values for each fold. If values are missing
            in folds, they will be added  through this function.
            features_per_fold (list): The features that were used for training in
            each fold.
            data (pd.DataFrame): The data that was used for training and validation.

        Returns:
            list[list]: The data transformed through the preprocessor and the features
            that are present in this data.
        """
        preprocessor = HancockTabularPreprocessor(
            data.columns[2:], min_max_scaler=True)
        preprocessor.set_output(transform="pandas")
        data_preprocessed = preprocessor.fit_transform(
            data.drop(["patient_id", "target"], axis=1))
        feature_names = preprocessor.get_feature_names_out()
        for i in range(len(shap_values)):
            features = features_per_fold[i]
            missing = set(feature_names) - set(features)
            for m in missing:
                # Find index of missing category
                missing_idx = list(feature_names).index(m)
                # Insert zeros for missing category
                shap_values[i] = np.insert(
                    shap_values[i], missing_idx, 0, axis=1)
        return [data_preprocessed, feature_names]


class AbstractHancockPredictorLime(AbstractHancockPredictor):
    """
    Abstract Class for the Prediction using the Hancock Data.

    There are already basic routines implemented for the cross_validation, train
    and predict methods. It will use the LIME method on the trainings data set to visualize
    the feature importance. It also overwrites the preprocess_data_for_model method to
    create for the labels a one-hot encoded array to work better with DNNs.

    To implement:
        - _prepare_data: Data getting process.
        - _prepare_data_split: Getting the correct data split for the training and validation data.
        - _get_df_train: Use the data split and data to return the training data.
        - _get_df_test: Use the data split and data to return the test data.
        - _create_new_model: Create a new model for the prediction process.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    # ================================================================================================================
    # Train
    # ================================================================================================================
    def train(
            self, df_train: pd.DataFrame = None, df_other: pd.DataFrame = None,
            plot_name: str = 'attention_adjuvant_treatment', model_reset: bool = True,
            batch_size: int = 32, epochs: int = 10, lime_flag: bool = False, **kwargs
    ) -> list:
        """This method trains the model on the given data and returns
        performance metrics for the validation data as well as the
        data that was used for training and validation. It additionally
        plots a LIME plot if lime_flag is set to True to visualize the
        feature importance.

        Args:
             df_train (pd.DataFrame, optional): The data that should be used
             to train the model. Should have the columns 'patient_id' and 'target'.
             Defaults to None and then the self.df_train property is used.

             df_other (pd.DataFrame, optional): The data that should be used to validate
             the trainings results. Should have the columns 'patient_id' and 'target'.

             Defaults to None and then the self.df_test property.
             plot_name (str, optional): The name of the plot that should be saved.

             model_reset (bool, optional): If this is set to True the model will be
             reset to None before the training process. Defaults to True.

             batch_size (int, optional): The batch size for the training process.
             Defaults to 32.

             epochs (int, optional): The number of epochs for the training process.

            lime_flag (bool, optional): If this is set to True the LIME plot will be
            created. Defaults to True.

        Returns:
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
        df_train, df_other = self._setup_training(df_train, df_other, model_reset)

        [[x_train, y_train, x_other, y_other], features] = self.prepare_data_for_model(
            df_train, df_other)

        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_data=(x_other, y_other))

        y_pred = self.predict(x_other)
        scores = self._training_calculate_metrics(y_other, y_pred)

        if self._plotter.plot_flag or self._plotter.save_flag:
            self._plot_train(df_other, y_pred, plot_name)
            if lime_flag:
                self._plot_lime(x_train, x_other, y_other, y_pred, features)

        return [scores, [x_train, y_train, x_other, y_other], features]

    def _plot_lime(
            self, x_train: np.ndarray, x_other: np.ndarray,
            y_other: np.ndarray, y_pred: np.array, features: np.ndarray,
            plot_name: str = 'attention_adjuvant_treatment_lime') -> None:
        """Creates the LIME plot with the help of the self._plotter.
        The plot is a summarization of the single lime plots for each
        misclassified sample.
        It will be considered the top 10 features for each sample. In the final
        plot only the 10 features that were most often in the top 10 features
        of the single samples are shown. This function is rather slow and should
        be used with caution if a large x_other is used, because for each sample
        an individual explainer has to be fitted.

        Args:
            x_train (np.ndarray): The training data already processed with
            the prepare_data_for_model method.

            x_other (np.ndarray): The test or validation data already processed with
            the prepare_data_for_model method.

            y_other (np.ndarray): The ground truth labels for the x_other data.
            Should be a one-hot encoded array for two classes, as it is returned
            from the prepare_data_for_model method.

            y_pred (np.array): The predicted labels for the x_other data. Should
            be only the predictions for class 1 as it is returned by the predict
            method.

            features (np.ndarray): The feature names of the x_other data. Should
            be the same as the ones returned from the prepare_data_for_model method.

            plot_name (str, optional): The name of the plot that should be saved.
        """
        n_features = 10
        explainer = lime_tabular.LimeTabularExplainer(
            x_train, feature_names=features, class_names=['class_0', 'class_1'],
            verbose=True, mode='classification', random_state=self._random_number
        )
        feature_names = []
        feature_values = []

        for index, (ground_truth, prediction, data_element) in enumerate(
                zip(y_other[:, 1], y_pred, x_other)):
            prediction = (prediction > 0.5).astype(int)
            if ground_truth != prediction:
                i = index
                exp = explainer.explain_instance(x_other[i], self.model.predict, num_features=10)
                exp_list = exp.as_list()
                for item in exp_list:
                    name_split = item[0].split(' ')
                    feature_names.append(name_split[0])
                    feature_values.append(item[1])

        feature_values = np.array(feature_values)
        feature_names = np.array(feature_names)
        self._plotter.lime_plot(
            feature_names, feature_values, n_features,
            colormap='coolwarm', plot_name=plot_name
        )

    # ================================================================================================================
    # Predict
    # ================================================================================================================
    def predict(self, data: pd.DataFrame | np.ndarray) -> np.array:
        y_pred = self.model.predict(data)[:, 1]
        return y_pred

    # ===============================================================================================================
    # Useful Helper Methods
    # ===============================================================================================================
    def prepare_data_for_model(
            self, df_train_fold: pd.DataFrame, df_other_fold: pd.DataFrame
    ) -> list:
        """
        Preprocess the input data and creates a numpy array for the labels.
        Also, over-samples the underrepresented class if necessary.

        Args:
            df_train_fold (pd.DataFrame): The data intended for training. Must have
            a column 'patient_id' and 'target'.
            df_other_fold (pd.DataFrame): The data intended for validating or testing the training process.

        Returns:
            list: List with [[x_train, y_train, x_other, y_other], features] in that order.
        """
        [[x_train, y_train, x_other, y_other], features] = super().prepare_data_for_model(
            df_train_fold, df_other_fold
        )
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
        y_other = tf.keras.utils.to_categorical(y_other, num_classes=2)
        x_train = x_train.astype(np.float32)
        x_other = x_other.astype(np.float32)
        return [[x_train, y_train, x_other, y_other], features]
