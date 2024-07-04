import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lifelines import KaplanMeierFitter, statistics
from matplotlib import rcParams

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline
from data_reader import DataFrameReaderFactory
from argument_parser import HancockArgumentParser
from defaults import DefaultFileNames


def training_and_testing(dataframe, dataframe_test, random_state):
    # Preprocess data
    preprocessor = setup_preprocessing_pipeline(
        dataframe.columns[2:], min_max_scaler=True)

    x_train = preprocessor.fit_transform(
        dataframe.drop(["patient_id", "target"], axis=1))
    x_test = preprocessor.transform(
        dataframe_test.drop(["patient_id", "target"], axis=1))
    y_train = dataframe["target"].to_numpy()
    y_test = dataframe_test["target"].to_numpy()

    # Handle class imbalance
    smote = SMOTE(random_state=random_state)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Fit ML model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)

    # Get predictions for test dataset
    y_pred = model.predict_proba(x_test)[:, 1]

    print(classification_report(y_test, y_pred > 0.5))

    # Threshold predictions
    y_pred = (y_pred > 0.5).astype(int)

    # Load survival data for test cases
    os_rfs = pd.read_csv(features_dir/"targets.csv", dtype={"patient_id": str})
    os_rfs = os_rfs[os_rfs.patient_id.isin(
        dataframe_test.patient_id.tolist())].reset_index(drop=True)
    # Define event: 1 = deceased, 0 = censored
    os_rfs.survival_status = os_rfs.survival_status == "deceased"
    os_rfs.survival_status = os_rfs.survival_status.astype(int)
    # Define event: 1 = recurrence, 0 = censored
    os_rfs.recurrence = os_rfs.recurrence == "yes"
    os_rfs.recurrence = os_rfs.recurrence.astype(int)
    # Convert number of days to number of months
    avg_days_per_month = 365.25/12
    os_rfs["followup_months"] = np.round(
        os_rfs["days_to_last_information"]/avg_days_per_month)
    os_rfs["months_to_rfs_event"] = np.round(
        os_rfs["days_to_rfs_event"]/avg_days_per_month)

    # Split cases by predictions
    idx0 = np.argwhere(y_pred == 0).flatten()
    idx1 = np.argwhere(y_pred == 1).flatten()
    pred_0 = os_rfs.iloc[idx0].reset_index(drop=True)
    pred_1 = os_rfs.iloc[idx1].reset_index(drop=True)

    # Log-rank test (survival)
    log_rank_result_os = statistics.logrank_test(
        durations_A=pred_0.followup_months,
        durations_B=pred_1.followup_months,
        event_observed_A=pred_0.survival_status,
        event_observed_B=pred_1.survival_status
    )
    p_os = log_rank_result_os.p_value
    p_value_os = get_significance(p_os)

    # Overall survival, grouped by predictions
    plt.figure(figsize=(2.5, 1.4))

    kmf0 = KaplanMeierFitter()
    kmf0.fit(pred_0.followup_months, pred_0.survival_status,
             label="pred. no adjuvant therapy")
    ax = kmf0.plot_survival_function(
        ci_show=True, show_censors=False, color=sns.color_palette("Set2")[0], linewidth=1)

    kmf1 = KaplanMeierFitter()
    kmf1.fit(pred_1.followup_months, pred_1.survival_status,
             label="pred. adjuvant therapy")
    kmf1.plot_survival_function(
        ci_show=True, show_censors=False, color=sns.color_palette("Set2")[1], linewidth=1)

    plt.text(50, 0.9, p_value_os)
    plt.ylabel("Overall survival", fontsize=6)
    plt.xlabel("Time since diagnosis [months]", fontsize=6)
    plt.xlim([0, ax.get_xticks()[-2]])
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.5))
    plt.savefig(results_dir/"adjuvant_treatment_prediction_os_test.svg",
                bbox_inches="tight")
    plt.close()

    # Log-rank test (Recurrence-free survival, grouped by predictions)
    log_rank_result_os = statistics.logrank_test(
        durations_A=pred_0.months_to_rfs_event,
        durations_B=pred_1.months_to_rfs_event,
        event_observed_A=pred_0.rfs_event,
        event_observed_B=pred_1.rfs_event
    )
    p_rfs = log_rank_result_os.p_value
    p_value_rfs = get_significance(p_rfs)

    plt.figure(figsize=(2.5, 1.4))

    kmf0 = KaplanMeierFitter()
    kmf0.fit(pred_0.months_to_rfs_event, pred_0.rfs_event,
             label="pred. no adjuvant therapy")
    ax = kmf0.plot_survival_function(
        ci_show=True, show_censors=False, color=sns.color_palette("Set2")[0], linewidth=1)

    kmf1 = KaplanMeierFitter()
    kmf1.fit(pred_1.months_to_rfs_event, pred_1.rfs_event,
             label="pred. adjuvant therapy")
    kmf1.plot_survival_function(
        ci_show=True, show_censors=False, color=sns.color_palette("Set2")[1], linewidth=1)

    plt.text(50, 0.9, p_value_rfs)
    plt.ylabel("Recurrence-free survival", fontsize=6)
    plt.xlabel("Time since surgery [months]", fontsize=6)
    plt.xlim([0, ax.get_xticks()[-2]])
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.5))
    plt.savefig(results_dir/"adjuvant_treatment_prediction_rfs_test.svg",
                bbox_inches="tight")
    plt.close()

    # Bar plot showing predictions for test dataset
    plt.figure(figsize=(3, 0.8))
    num_pred_adjuvant = np.count_nonzero(y_pred)
    num_pred_none = np.count_nonzero(y_pred == 0)
    ax = plt.barh([0, 1], [num_pred_none, num_pred_adjuvant],
                  color=sns.color_palette("Set2")[1])
    plt.bar_label(ax, fontsize=6, padding=2)
    plt.yticks(ticks=[0, 1], labels=[
               "Predicted no adjuvant therapy", "Predicted adjuvant therapy"])
    plt.xticks([0, 20, 40, 60, 80])
    plt.xlabel("# Patients", fontsize=6)
    plt.tight_layout()
    sns.despine()
    plt.savefig(
        results_dir/"adjuvant_treatment_prediction_bar_plot_test.svg", bbox_inches="tight")
    plt.close()


def get_significance(p_value) -> str:
    if p_value <= 0.001:
        return "$p\\leq$0.001 (***)"
    elif p_value <= 0.01:
        return "$p\\leq$0.01 (**)"
    elif p_value <= 0.05:
        return "$p\\leq$0.05 (*)"
    else:
        return f"$p=${p_value: .3f}"


class AbstractHancockPredictor:
    """Abstract class that defines the interface for all predictors in the Hancock
    project. All predictors should inherit from this class. Do not initialize
    this class directly, as it will throw a NotImplementedError.
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
            self.model_setter_flag = False
            self._model = self._get_model()
        return self._model

    @model.setter
    def model(self, model):
        """Setter for the model property. If the setter is used, the model
        will no longer be initialized by the _get_model method and thus
        the training process will be continued with the given model. 
        If this process should be discarded and the initial model should be 
        used, change first the model to None and afterward the
        self.model_setter_flag to False.

        Args:
            model: The model that should be used to make the predictions.
        """
        self.model_setter_flag = True
        self._model = model

    def __init__(
        self, save_flag: bool = False, plot_flag: bool = False,
        random_state: int = 42, predictor_type: str = 'None'
    ):
        """Creates a new HancockPredictor object. The object should be used
        to train, test and predict the HANCOCK data.

        Args:
            save_flag (bool, optional): If this is set to true the generated 
            plots will be saved. Defaults to False.
            plot_flag (bool, optional): If this is set to true the generated 
            plots are displayed to the user. Defaults to False.
            random_state (int, optional): To make sure that the outputs are
            reproducible we initialize all processes that involve randomness 
            with the random_state. Defaults to 42.
            predictor_type (str, optional): The type of the predictor. 
            Is used to initialize the HancockArgumentParser. Defaults to 'None'.
        """
        rcParams.update({"font.size": 6})
        rcParams["svg.fonttype"] = "none"
        self.argumentParser = HancockArgumentParser(
            type=predictor_type)
        self.args = self.argumentParser.parse_args()
        self.save_flag = save_flag
        self.plot_flag = plot_flag
        self.random_state = np.random.RandomState(random_state)
        self._data_reader_factory = DataFrameReaderFactory()
        self._default_file_names = DefaultFileNames()

        self._df_train = None
        self._df_test = None
        self._model = None
        self.model_setter_flag = False

        self._prepare_data()
        self._prepare_data_split()

    def cross_validate(
        self, n_splits: int = 10, plot_name: str = 'cross_validate',
    ) -> None:
        raise NotImplementedError("Cross-validation not implemented.")

    def train(self, df_train: pd.DataFrame = None, df_val: pd.DataFrame = None, plot_name: str = 'train',
              ) -> list:
        raise NotImplementedError("Training not implemented.")

    def test(self, df: pd.DataFrame = None, plot_name: str = 'test',
             ) -> None:
        raise NotImplementedError("Testing not implemented.")

    def predict(self, data: pd.DataFrame) -> np.array:
        raise NotImplementedError("Prediction not implemented.")

    def _prepare_data(self) -> None:
        """Should implement the data getting process and write the data to 
        the self._data property. The data should be a pandas data frame.

        Raises:
            NotImplementedError: If the method is not implemented by the 
            child class.
        """
        self._data = None
        raise NotImplementedError("Data preparing not implemented")

    def _prepare_data_split(self) -> None:
        """Prepares the self._data_split property for the _get_df_train and 
        _get_df_test methods. 

        Raises:
            NotImplementedError: if the method is not implemented by the 
            child class.
        """
        self._data_split = None
        raise NotImplementedError("Data split getting not implemented.")

    def _get_df_train(self) -> pd.DataFrame:
        raise NotImplementedError("Data frame for training not implemented.")

    def _get_df_test(self) -> pd.DataFrame:
        raise NotImplementedError("Data frame for testing not implemented.")

    def _get_model(self):
        if self.model_setter_flag:
            return self._model
        else:
            return self._create_new_model()

    def _create_new_model(self):
        raise NotImplementedError("Model creation not implemented.")


class TabularAdjuvantTreatmentPredictor(AbstractHancockPredictor):
    def __init__(
        self, save_flag: bool = False, plot_flag: bool = False,
        random_state: int = 42
    ):
        super().__init__(save_flag=save_flag, plot_flag=plot_flag,
                         random_state=random_state,
                         predictor_type='adjuvant_treatment_prediction'
                         )

    # ----- Cross validation -----
    def cross_validate(
        self, n_splits: int = 10, plot_name: str = 'multimodal',
    ) -> list[list]:
        """Performs cross-validation on the training data (df_train) with
        n_split folds. The cross-validation is done with a StratifiedKFold.

        Args:
            n_splits (int): The number of splits for the cross-validation.
            Defaults to 10.
            plot_name (str): The name of the plot that should be saved to disk.

        Returns:
            list[list]: A list with [tpr_list, auc_list, shap_values, val_index_list,
            features_per_fold]
            tpr_list: List with the true positive rates for each fold.
            auc_list: List with the AUC scores for each fold.
            shap_values: List with the SHAP values for each fold.
            val_index_list: List with the indices of the validation data for each fold.
            features_per_fold: List with the ColumnTransformer features for each fold.
        """
        tpr_list = []
        auc_list = []
        shap_values = []
        val_index_list = []
        features_per_fold = []

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=self.random_state)
        for train_idx, val_idx in cv.split(self.df_train, self.df_train['target']):
            val_index_list.append(val_idx)
            df_train_fold = self.df_train.iloc[train_idx]
            df_val_fold = self.df_train.iloc[val_idx]
            self._cross_validation_single_fold(
                df_train_fold, df_val_fold, tpr_list, auc_list, shap_values,
                features_per_fold, plot_name
            )

        self._cross_validation_shap(shap_values, features_per_fold, val_index_list,
                                    self.df_train, plot_name)

        return [tpr_list, auc_list, shap_values, val_index_list, features_per_fold]

    def _cross_validation_single_fold(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame,
            tpr_list: list, auc_list: list, shap_values: list,
            features_per_fold: list, plot_name: str
    ):
        """Performs the training for a single fold and adds the results to the
        corresponding lists. Also resets the self._model property to None.

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
        train_return = self.train(
            df_train=df_train_fold, df_val=df_val_fold, plot_name=plot_name
        )
        tpr = train_return[0][1]
        tpr_list.append(tpr)

        roc_score = train_return[0][2]
        auc_list.append(roc_score)

        features = train_return[2]
        features_per_fold.append(features)

        x_val = train_return[1][2]
        svs = shap.TreeExplainer(self.model).shap_values(x_val)
        shap_values.append(svs[:, :, 1])  # class = 1 (adjuvant treatment yes)

        self._model = None

    def _cross_validation_shap(
            self, shap_values: list, features_per_fold: list,
            val_index_list: list, data: pd.DataFrame, plot_name: str
    ):
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

        self._cross_validation_plot_shap_values(
            shap_values, data_preprocessed, feature_names, all_val_folds_idx
        )
        self._cross_validation_save_shap_values(
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
        preprocessor = setup_preprocessing_pipeline(
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
                shap_values[i] = np.insert(shap_values[i], missing_idx, 0, axis=1)
        return [data_preprocessed, feature_names]

    def _cross_validation_plot_shap_values(
            self, shap_values: list, data_preprocessed: pd.DataFrame,
            feature_names: list, all_val_folds_idx: list
    ):
        """Plots the SHAP values for the cross-validation if the self.plot_flag == True.

        Args:
            shap_values (list): The SHAP values for each fold.
            data_preprocessed (pd.DataFrame): The data that was used for training and
            validation transformed through the preprocessor.
            feature_names (list): The features that were used for training.
            all_val_folds_idx (list): List of all indices that were used for validation in
            the data_preprocessed data frame.
        """
        shap.summary_plot(
            shap_values=np.concatenate(shap_values),
            features=data_preprocessed.reindex(all_val_folds_idx),
            feature_names=feature_names,
            max_display=12,
            show=self.plot_flag
        )

    def _cross_validation_save_shap_values(
            self, shap_values: list, data_preprocessed: pd.DataFrame,
            feature_names: list, all_val_folds_idx: list, plot_name: str
    ):
        """Saves the SHAP values for the cross-validation if the self.save_flag == True.
        The file name is 'shap_summary_plot_{plot_name}.svg'.

        Args:
            shap_values (list): The SHAP values for each fold.
            data_preprocessed (pd.DataFrame): The data that was used for training and
            validation transformed through the preprocessor.
            feature_names (list): The features that were used for training.
            all_val_folds_idx (list): List of all indices that were used for validation in
            the data_preprocessed data frame.
            plot_name (str): The name of the plot that should be saved.
        """
        if self.plot_flag:
            shap.summary_plot(
                shap_values=np.concatenate(shap_values),
                features=data_preprocessed.reindex(all_val_folds_idx),
                feature_names=feature_names,
                max_display=12,
                show=False
            )

        fig, ax = plt.gcf(), plt.gca()
        ax.tick_params(labelsize=8)
        ax.set_xlabel("SHAP value", fontsize=8)
        cbar = fig.axes[-1]
        cbar.tick_params(labelsize=8)
        fig.set_size_inches(6, 3)
        plt.tight_layout()
        if self.save_flag:
            plt.savefig(
                self.args.results_dir / f"shap_summary_plot_{plot_name}.svg",
                bbox_inches="tight"
            )
            plt.savefig(
                self.args.results_dir / f"shap_summary_plot_{plot_name}.png",
                bbox_inches="tight", dpi=200
            )
        plt.close()

    # ----- Training -----
    def train(
        self, df_train: pd.DataFrame = None, df_val: pd.DataFrame = None,
        plot_name: str = 'train',
    ) -> list:
        """This method trains the model on the given data and returns
        performance metrics for the validation data as well as the
        data that was used for training and validation.

        Args:
             df_train (pd.DataFrame, optional): The data that should be used
             to train the model. Should have the columns 'patient_id' and 'target'.
             Defaults to None and then the self.df_train property is used.

             df_val (pd.DataFrame, optional): The data that should be used to validate
             the trainings results. Should have the columns 'patient_id' and 'target'.
             Defaults to None and then the self.df_test property.
             plot_name (str, optional): The name of the plot that should be saved.

        Returns:
            list: A list with the validation parameters and the data that was used.
            [[fpr, tpr, auc], [x_train, y_train, x_val, y_val, y_pred], features]
            fpr: False positive rate
            tpr: True positive rate
            x_train: The training data encoded
            y_train: The training labels
            x_val: The validation data encoded
            y_val: The validation labels
            y_pred: The predicted labels for the validation data
            features: The features from the ColumnTransformer that was used for
            encoding the training data.
        """
        x_linspace = np.linspace(0, 1, 100)
        if df_train is None:
            df_train = self.df_train
        if df_val is None:
            df_val = self.df_test
        [[x_train, y_train, x_val, y_val], features] = self._prepare_data_for_training(
            df_train, df_val)
        self.model.fit(x_train, y_train)
        y_pred = self.predict(x_val)
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        tpr = np.interp(x_linspace, fpr, tpr)
        tpr[0] = 0.0
        tpr[-1] = 1.0
        print(
            f'Training Classification Report: \n{classification_report(y_val, y_pred > 0.5)}',
            end='\n\n'
        )
        return [[fpr, tpr, roc_auc_score(y_val, y_pred)],
                [x_train, y_train, x_val, y_val, y_pred], features]

    def _prepare_data_for_training(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame
    ) -> list:
        """Preprocess the input data and creates a numpy array for the labels.
        Also, over-samples the underrepresented class if necessary.

        Args:
            df_train_fold (pd.DataFrame): The data intended for training. Must have
            a column 'patient_id' and 'target'.
            df_val_fold (pd.DataFrame): The data intended for validating the training process.

        Returns:
            list: List with [[x_train, y_train, x_val, y_val], features] in that order.
        """
        preprocessor = setup_preprocessing_pipeline(
            df_train_fold.columns[2:], min_max_scaler=True)

        x_train = preprocessor.fit_transform(
            df_train_fold.drop(["patient_id", "target"], axis=1))
        x_val = preprocessor.transform(
            df_val_fold.drop(["patient_id", "target"], axis=1))
        y_train = df_train_fold["target"].to_numpy()
        y_val = df_val_fold["target"].to_numpy()

        features = preprocessor.get_feature_names_out()

        # Handle class imbalance
        smote = SMOTE(random_state=self.random_state)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        return [[x_train, y_train, x_val, y_val], features]

    # ---- Prediction ----
    def predict(self, data: pd.DataFrame) -> np.array:
        y_pred = self.model.predict_proba(data)[:, 1]
        return y_pred

    # ---- Data preparation ----
    def _prepare_data(self) -> None:
        """Prepares the self._data property for the _get_df_train and _get_df_test
        methods. For this predictor the StructuralAggregated data is used.
        """
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.structural_aggregated,
            data_dir=self.args.features_dir,
            data_dir_flag=True
        )
        self._data = data_reader.return_data()

    def _prepare_data_split(self) -> None:
        """Prepares the self._data_split property for the _get_df_train and 
        _get_df_test methods. For this predictor the treatment outcome data 
        split is used. 
        """
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_treatment_outcome,
            data_dir=self.args.data_split_dir /
            self._default_file_names.data_split_treatment_outcome,
            data_dir_flag=True
        )
        self._data_split = data_split_reader.return_data()
        self._data_split["target"] = self._data_split["adjuvant_treatment"].apply(
            lambda x: 0 if x == "none" else 1)

    def _get_df_train(self) -> pd.DataFrame:
        target_train = self._data_split[self._data_split.dataset ==
                                        "training"][["patient_id", "target"]]
        return target_train.merge(self._data, on="patient_id", how="inner")

    def _get_df_test(self) -> pd.DataFrame:
        target_test = self._data_split[
            self._data_split.dataset == "test"][["patient_id", "target"]]
        return target_test.merge(self._data, on="patient_id", how="inner")

    # ---- Model creation ----
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(n_estimators=800, random_state=self.random_state)
        return model


if __name__ == "__main__":
    predictor = TabularAdjuvantTreatmentPredictor(save_flag=True, plot_flag=True)
    cross_validation_splits = 10

    data_split_dir = predictor.args.data_split_dir
    results_dir = predictor.args.results_dir
    features_dir = predictor.args.features_dir

    df_train_to_delete = predictor.df_train
    df_test_to_delete = predictor.df_test

    # Train classifier on multimodal data with k-fold CV
    print("Running k-fold cross-validation for multimodal data...")
    multi_modal_cross_validation = predictor.cross_validate(
        n_splits=cross_validation_splits, plot_name='multimodal_test')
    roc_multimodal = multi_modal_cross_validation[0]
    auc_multimodal = multi_modal_cross_validation[1]
    shap_values_multimodal = multi_modal_cross_validation[2]

    # # Train classifiers on single modalities with 10-fold CV
    # print("Running k-fold cross-validation for clinical data...")
    # roc_clinical, auc_clinical, _ = cross_validation(
    #     df_train[["target"] + list(clinical.columns)], predictor.rng, k=10)
    # print("Running k-fold cross-validation for pathological data...")
    # roc_patho, auc_patho, _ = cross_validation(
    #     df_train[["target"] + list(patho.columns)], predictor.rng, k=10)
    # print("Running k-fold cross-validation for blood data...")
    # roc_blood, auc_blood, _ = cross_validation(
    #     df_train[["target"] + list(blood.columns)], predictor.rng, k=10)
    # print("Running k-fold cross-validation for cell density data...")
    # roc_tma, auc_tma, _ = cross_validation(
    #     df_train[["target"] + list(cell_density.columns)], predictor.rng, k=10)
    # print("Running k-fold cross-validation for text data...")
    # roc_icd, auc_icd, _ = cross_validation(
    #     df_train[["target"] + list(icd.columns)], predictor.rng, k=10)

    # # Plot average ROC curves with AUC scores
    # colors = [(132/255, 163/255, 204/255)] * 6
    # auc_list = [auc_multimodal, auc_clinical,
    #             auc_patho, auc_blood, auc_tma, auc_icd]
    # roc_list = [roc_multimodal, roc_clinical,
    #             roc_patho, roc_blood, roc_tma, roc_icd]
    # roc_labels = ["Multimodal", "Clinical", "Pathology",
    #               "Blood", "TMA cell density", "ICD codes"]

    # colors = [(132/255, 163/255, 204/255)]
    # auc_list = [auc_multimodal]
    # roc_list = [roc_multimodal]
    # roc_labels = ['Multimodal']
    #
    # for i in range(len(colors)):
    #     plt.figure(figsize=(1.4, 1.4))  # plt.figure(figsize=(1, 2.5))
    #     mean_tpr = np.mean(roc_list[i], axis=0)
    #     std_tpr = np.std(roc_list[i], axis=0)
    #     tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    #     tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    #     mean_fpr = np.linspace(0, 1, 100)
    #
    #     plt.plot(mean_fpr, mean_tpr, linewidth=1,
    #              color=colors[i], label=f"AUC =\n{np.mean(auc_list[i]):.2f}")
    #     plt.fill_between(mean_fpr, tpr_lower, tpr_upper,
    #                      label=r"$\pm$ std.", color=colors[i], alpha=0.4, lw=0)
    #
    #     plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1)  # random
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.xlabel("FPR", fontsize=6)
    #     plt.ylabel("TPR", fontsize=6)
    #     plt.title(f"{roc_labels[i]}")
    #     # plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    #     plt.legend(frameon=False, loc="lower right", borderpad=0)
    #     plt.tight_layout()
    #     plt.gca().set_aspect("equal")
    #     plt.savefig(
    #         results_dir/f"roc_treatment_{roc_labels[i]}_test.svg", bbox_inches="tight")
    #     plt.close()

    # # Train classifier once on multimodal data, show survival curves and bar plot
    # print("Training and testing the final multimodal model...")
    # training_and_testing(df_train, df_test, predictor.rng)
    # print(f"Done. Saved results to {results_dir}")
