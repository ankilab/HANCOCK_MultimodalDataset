import warnings
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
import copy
import tensorflow as tf
from collections import Counter
from lime import lime_tabular

from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning.custom_preprocessor import ColumnPreprocessor
from multimodal_machine_learning.custom_preprocessor import TMA_VECTOR_LENGTH, TMA_VECTOR_FEATURES
from data_reader import DataFrameReaderFactory
from argument_parser import HancockArgumentParser
from defaults import DefaultFileNames
from multimodal_machine_learning.attention_model import AttentionMLPModel


def get_significance(p_value) -> str:
    if p_value <= 0.001:
        return "$p\\leq$0.001 (***)"
    elif p_value <= 0.01:
        return "$p\\leq$0.01 (**)"
    elif p_value <= 0.05:
        return "$p\\leq$0.05 (*)"
    else:
        return f"$p=${p_value: .3f}"


class PredictionPlotter:
    def __init__(self, save_dir: Path, save_flag: bool = True, plot_flag: bool = True):
        self._backup_save_flag = None
        self._backup_plot_flag = None
        self.save_dir = save_dir
        self.save_flag = save_flag
        self.plot_flag = plot_flag

    def deactivate(self):
        """Saves the status of the plot and save flag and sets it to False.
        """
        self._backup_save_flag = self.save_flag
        self._backup_plot_flag = self.plot_flag
        self.save_flag = False
        self.plot_flag = False

    def reactivate(self):
        """Reads the backup status of the plot and save flag from the last
        deactivate call and sets it to the current status.
        """
        if self._backup_plot_flag is None or self._backup_plot_flag is None:
            return
        self.save_flag = self._backup_save_flag
        self.plot_flag = self._backup_plot_flag

    def prediction_plot(
            self, y_pred: np.array, plot_name: str = 'adjuvant_treatment'
    ) -> None:
        plt.figure(figsize=(3, 0.8))
        num_pred_adjuvant = np.count_nonzero(y_pred)
        num_pred_none = np.count_nonzero(y_pred == 0)
        ax = plt.barh([0, 1],
                      [num_pred_none, num_pred_adjuvant], color=sns.color_palette("Set2")[1])
        plt.bar_label(ax, fontsize=6, padding=2)
        plt.yticks(
            ticks=[0, 1],
            labels=["Predicted no adjuvant therapy",
                    "Predicted adjuvant therapy"]
        )
        plt.xticks([0, 20, 40, 60, 80])
        plt.xlabel("# Patients", fontsize=6)
        plt.tight_layout()
        sns.despine()
        if self.save_flag:
            plt.savefig(
                self.save_dir / f"{plot_name}_prediction_bar_plot.svg", bbox_inches="tight")
            plt.savefig(
                self.save_dir / f"{plot_name}_prediction_bar_plot.png", bbox_inches="tight")

        if self.plot_flag:
            plt.show()
        else:
            plt.close()

    def follow_up_months_plot(
            self, survival_targets: pd.DataFrame, y_pred: np.array,
            plot_name: str = 'adjuvant_treatment'
    ) -> None:
        idx0 = np.argwhere(y_pred == 0).flatten()
        idx1 = np.argwhere(y_pred == 1).flatten()
        survival_targets_0_prediction = survival_targets.iloc[idx0].reset_index(
            drop=True)
        survival_targets_1_prediction = survival_targets.iloc[idx1].reset_index(
            drop=True)

        log_rank_result_os = statistics.logrank_test(
            durations_A=survival_targets_0_prediction.followup_months,
            durations_B=survival_targets_1_prediction.followup_months,
            event_observed_A=survival_targets_0_prediction.survival_status,
            event_observed_B=survival_targets_1_prediction.survival_status
        )
        p_os = log_rank_result_os.p_value
        p_value_os = get_significance(p_os)

        # Overall survival, grouped by predictions
        plt.figure(figsize=(2.5, 1.4))
        kmf0 = KaplanMeierFitter()
        kmf0.fit(
            survival_targets_0_prediction.followup_months, survival_targets_0_prediction.survival_status,
            label="pred. no adjuvant therapy"
        )
        ax = kmf0.plot_survival_function(
            ci_show=True, show_censors=False, color=sns.color_palette("Set2")[0], linewidth=1)
        kmf1 = KaplanMeierFitter()
        kmf1.fit(
            survival_targets_1_prediction.followup_months, survival_targets_1_prediction.survival_status,
            label="pred. adjuvant therapy")
        kmf1.plot_survival_function(
            ci_show=True, show_censors=False, color=sns.color_palette("Set2")[1], linewidth=1
        )

        plt.text(50, 0.9, p_value_os)
        plt.ylabel("Overall survival", fontsize=6)
        plt.xlabel("Time since diagnosis [months]", fontsize=6)
        plt.xlim([0, ax.get_xticks()[-2]])
        plt.tight_layout()
        sns.despine()
        plt.legend(frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, -0.5))

        if self.save_flag:
            plt.savefig(self.save_dir / f"{plot_name}_prediction_os.svg",
                        bbox_inches="tight")

        if self.plot_flag:
            plt.show()
        else:
            plt.close()

    def recurrence_free_survival_plot(
            self, survival_targets: pd.DataFrame, y_pred: np.array,
            plot_name: str = 'adjuvant_treatment'
    ) -> None:
        idx0 = np.argwhere(y_pred == 0).flatten()
        idx1 = np.argwhere(y_pred == 1).flatten()
        survival_targets_0_prediction = survival_targets.iloc[idx0].reset_index(
            drop=True)
        survival_targets_1_prediction = survival_targets.iloc[idx1].reset_index(
            drop=True)

        # Log-rank test (Recurrence-free survival, grouped by predictions)
        log_rank_result_os = statistics.logrank_test(
            durations_A=survival_targets_0_prediction.months_to_rfs_event,
            durations_B=survival_targets_1_prediction.months_to_rfs_event,
            event_observed_A=survival_targets_0_prediction.rfs_event,
            event_observed_B=survival_targets_1_prediction.rfs_event
        )
        p_rfs = log_rank_result_os.p_value
        p_value_rfs = get_significance(p_rfs)

        plt.figure(figsize=(2.5, 1.4))
        kmf0 = KaplanMeierFitter()
        kmf0.fit(
            survival_targets_0_prediction.months_to_rfs_event, survival_targets_0_prediction.rfs_event,
            label="pred. no adjuvant therapy")
        ax = kmf0.plot_survival_function(
            ci_show=True, show_censors=False, color=sns.color_palette("Set2")[0], linewidth=1)
        kmf1 = KaplanMeierFitter()
        kmf1.fit(
            survival_targets_1_prediction.months_to_rfs_event, survival_targets_1_prediction.rfs_event,
            label="pred. adjuvant therapy")
        kmf1.plot_survival_function(
            ci_show=True, show_censors=False, color=sns.color_palette("Set2")[1], linewidth=1)
        plt.text(50, 0.9, p_value_rfs)
        plt.ylabel("Recurrence-free survival", fontsize=6)
        plt.xlabel("Time since surgery [months]", fontsize=6)
        plt.xlim([0, ax.get_xticks()[-2]])
        plt.tight_layout()
        sns.despine()
        plt.legend(frameon=False, loc='upper center',
                   bbox_to_anchor=(0.5, -0.5))

        if self.save_flag:
            plt.savefig(self.save_dir / f"{plot_name}_prediction_rfs.svg",
                        bbox_inches="tight")
        if self.plot_flag:
            plt.show()
        else:
            plt.close()

    def plot_shap_values(
            self, shap_values: list, data_preprocessed: pd.DataFrame,
            feature_names: list, all_val_folds_idx: list
    ) -> None:
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
            shap_values=np.concatenate(shap_values, axis=0),
            features=data_preprocessed.reindex(all_val_folds_idx),
            feature_names=feature_names,
            max_display=12,
            show=self.plot_flag
        )

    def save_shap_values(
            self, shap_values: list, data_preprocessed: pd.DataFrame,
            feature_names: list, all_val_folds_idx: list, plot_name: str
    ) -> None:
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
                self.save_dir / f"shap_summary_plot_{plot_name}.svg",
                bbox_inches="tight"
            )
            plt.savefig(
                self.save_dir / f"shap_summary_plot_{plot_name}.png",
                bbox_inches="tight", dpi=200
            )
        plt.close()

    def roc_curve(
            self, auc_score: np.array, tpr_list: list,
            plot_name: str
    ) -> None:
        """Creates the ROC curve for the cross-validation if the self.plot_flag == True
        it will be shown to the user, if the self.save_flag is True it will be save to disk.

        Args:
            auc_score (np.array): The AUC scores for each fold.
            tpr_list (list): The true positive rates for each fold.
            plot_name (str): the name of the plot.
        """
        color = (132 / 255, 163 / 255, 204 / 255)
        plt.figure(figsize=(1.4, 1.4))
        mean_tpr = np.mean(tpr_list, axis=0)
        std_tpr = np.std(tpr_list, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_fpr = np.linspace(0, 1, 100)

        plt.plot(mean_fpr, mean_tpr, linewidth=1,
                 color=color, label=f"AUC =\n{np.mean(auc_score): .2f}")
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper,
                         label=r"$\pm$ std.", color=color, alpha=0.4, lw=0)

        plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1)  # random
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("FPR", fontsize=6)
        plt.ylabel("TPR", fontsize=6)
        plt.title(f"{plot_name}")
        # plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
        plt.legend(frameon=False, loc="lower right", borderpad=0)
        plt.tight_layout()
        plt.gca().set_aspect("equal")
        if self.save_flag:
            plt.savefig(
                self.save_dir / f"roc_treatment_{plot_name}.svg", bbox_inches="tight")
            plt.savefig(
                self.save_dir / f"roc_treatment_{plot_name}.png", bbox_inches="tight")
        if self.plot_flag:
            plt.show()
        else:
            plt.close()

    def predictor_comparison_table(
            self,
            metrics_data: list[list],
            metrics_labels: list[str],
            row_labels: list[str],
            fig_size: tuple[int, int] = (18, 6),
            plot_name: str = 'comparison_adjuvant_treatment'
    ) -> None:
        fig, ax = plt.subplots(figsize=fig_size)
        _ = ax.axis('tight')
        _ = ax.axis('off')
        the_table = ax.table(cellText=metrics_data, rowLabels=row_labels, colLabels=metrics_labels, loc='center')
        the_table.set_fontsize(10)
        the_table.auto_set_column_width(col=list(range(len(metrics_labels))))

        if self.save_flag:
            plt.savefig(self.save_dir / f'{plot_name}.png')
        if self.plot_flag:
            plt.show()
        else:
            plt.close()

    def lime_plot(
            self, feature_names: np.array, feature_values: np.array,
            n_features: int,
            colormap = 'coolwarm', fig_size = (8, 6), plot_name = 'lime_plot'
    ) -> None:
        """Creates a LIME summary plot for the top n_features most frequent features
        in the feature_names array.

        Args:
            feature_names (np.array): The names of the features. Should be matched to
            the feature_values array. Means it should have the same length.

            feature_values (np.array): The lime score values for the features. Should be
            matched to the feature_names array. Means it should have the same length.

            n_features (int): The number of most frequent features that should be shown
            in the LIME summary plot.

            colormap (str, optional): The seaborn colormap that should be used for the plot.
            Should be a diverging colormap. Defaults to 'coolwarm'.

            fig_size (tuple[int, int], optional): The size of the figure. Defaults to (8, 6).

            plot_name (str, optional): The save name of the plot. Defaults to 'lime_plot'.
        """
        feature_counts = Counter(feature_names)
        most_common_features = [feature for feature, _ in feature_counts.most_common(n_features)]
        filtered_feature_names = [name for name in feature_names if name in most_common_features]
        filtered_feature_values = [value for name, value in zip(feature_names, feature_values) if
                                   name in most_common_features]
        filtered_feature_names = np.array(filtered_feature_names)
        filtered_feature_values = np.array(filtered_feature_values)
        df = pd.DataFrame({
            'Feature Name': filtered_feature_names,
            'Feature Contribution Value': filtered_feature_values
        })

        palette = sns.color_palette(colormap, as_cmap=True)
        # Create scatter plot
        plt.figure(figsize=fig_size)
        scatter = sns.scatterplot(
            data=df,
            x='Feature Contribution Value',
            y='Feature Name',
            hue='Feature Contribution Value',
            palette=palette,
            s=100,  # Marker size
            alpha=0.7
        )

        plt.ylabel('Feature Name')
        plt.xlabel('Feature Contribution Value')
        plt.title(f'LIME Summary Plot - Top {n_features} Most Frequent Features')

        plt.tight_layout()

        if self.save_flag:
            plt.savefig(self.save_dir / f'{plot_name}.png')
            plt.savefig(self.save_dir / f'{plot_name}.svg')
        if self.plot_flag:
            plt.show()
        else:
            plt.close()


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
        self._save_flag = save_flag
        self._plot_flag = plot_flag
        self._plotter = PredictionPlotter(
            self.args.results_dir, save_flag, plot_flag)
        self._random_number = random_state
        self._data_reader_factory = DataFrameReaderFactory()
        self._default_file_names = DefaultFileNames()

        self._df_train = None
        self._df_test = None
        self._model = None
        self._untrained_model = None
        self.model_setter_flag = False

        self._prepare_data()
        self._prepare_data_split()

    def cross_validate(
        self, n_splits: int = 10, plot_name: str = 'cross_validate', **kwargs
    ) -> None:
        raise NotImplementedError("Cross-validation not implemented.")

    def train(self, df_train: pd.DataFrame = None, df_other: pd.DataFrame = None,
              plot_name: str = 'train', model_reset: bool = True, **kwargs
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
            if self._untrained_model is None:
                print('No model set. Either set a model or change the' +
                      ' model_setter_flag to False.')
            return copy.deepcopy(self._untrained_model)
        else:
            return self._create_new_model()

    def _create_new_model(self):
        raise NotImplementedError("Model creation not implemented.")


class AdjuvantTreatmentPredictor(AbstractHancockPredictor):
    """Class for predicting adjuvant treatments. Here the merged tabular
    data is used.
    """

    def __init__(
        self, save_flag: bool = False, plot_flag: bool = False,
        random_state: int = 42
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
        super().__init__(save_flag=save_flag, plot_flag=plot_flag,
                         random_state=random_state,
                         predictor_type='adjuvant_treatment_prediction'
                         )
        self._preprocessor = ColumnPreprocessor(
            self.df_train.columns[2:], min_max_scaler=True)
        self._preprocessor = self._preprocessor.fit(
            pd.concat([self.df_train, self.df_test]
                      ).drop(['patient_id', 'target'], axis=1))

    # ----- Cross validation -----
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
        self._cross_validation_shap(shap_values, features_per_fold, val_index_list,
                                    self.df_train, plot_name)
        self._plotter.roc_curve(auc_list, tpr_list, plot_name)

        return [tpr_list, auc_list, val_index_list, features_per_fold, shap_values]

    def _cross_validation_single_fold(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame,
            tpr_list: list, auc_list: list, shap_values: list,
            features_per_fold: list, plot_name: str
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

    def _cross_validation_single_fold_train_and_add_basic_metrics(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame, plot_name:str,
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
        preprocessor = ColumnPreprocessor(
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

    # ----- Training -----
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
            self, df_train:pd.DataFrame | None, df_other: pd.DataFrame | None, model_reset: bool
    )-> tuple[pd.DataFrame, pd.DataFrame]:
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
        self._plotter.follow_up_months_plot(
            survival_targets, y_pred, plot_name)
        self._plotter.recurrence_free_survival_plot(
            survival_targets, y_pred, plot_name)

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
            data_type=self._data_reader_factory.data_reader_types.tabular_merged_feature,
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
        target_train = target_train.merge(self._data, on="patient_id", how="inner")
        target_train = target_train[['target'] + target_train.columns.drop('target').to_list()]
        return target_train

    def _get_df_test(self) -> pd.DataFrame:
        target_test = self._data_split[
            self._data_split.dataset == "test"][["patient_id", "target"]]
        return target_test.merge(self._data, on="patient_id", how="inner")

    # ---- Model creation ----
    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=600, min_samples_split=5,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features='sqrt', max_depth=50,
            criterion='log_loss',
            random_state=np.random.RandomState(self._random_number)
        )
        return model

    # ----- General useful functions -----
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


class AbstractNeuralNetworkAdjuvantTreatmentPredictor(AdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments with a neural network.
    This is an abstract class and will throw a NotImplementedError if it is
    initialized directly.

    To implement:
        _create_new_model: Create a new model for the predictor and return it.
        It should be able to be called with fit and predict methods. So if
        needed compile it before the return.

        _prepare_data: Prepare the data for the df_tran and df_test properties.
        Use the _data_reader_factory to get the data and set it to the
        self._data attribute of the class.


    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    def __init__(self, save_flag: bool = False, plot_flag: bool = False, random_state: int = 42):
        """
        Initializes the NeuralNetworkAdjuvantTreatmentPredictor. It can be used to
        perform adjuvant treatment prediction with the merged tabular and tma data.

        Args:
            save_flag (bool, optional): If this is set to True the generated
            plots will be saved. Defaults to False.

            plot_flag (bool, optional): If this is set to True the generated
            plots will be shown to the user. Defaults to False.

            random_state (int, optional): To make sure that the outputs are
            reproducible we initialize all processes that involve randomness
            with the random_state. Defaults to 42.
        """
        super().__init__(save_flag=save_flag, plot_flag=plot_flag, random_state=random_state)
        [[x_train, y_train, x_val, y_val], features] = self.prepare_data_for_model(
            self.df_train, self.df_test
        )
        self._input_dim = x_train.shape[1]

    # ---- Cross Validation ----
    def cross_validate(
            self, n_splits: int = 10, plot_name: str = 'attention_adjuvant_treatment', **kwargs
    ) -> list:
        """Performs cross-validation on the training data (df_train) with
        n_split folds. The cross-validation is done with a StratifiedKFold.

        Args:
            n_splits (int): The number of splits for the cross-validation.
            Defaults to 10.
            plot_name (str): The name of the plot that should be saved to disk.
        """
        tpr_list = []
        auc_list = []
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
                df_train_fold, df_val_fold, tpr_list, auc_list, features_per_fold, plot_name
            )

        self._plotter.reactivate()
        self._plotter.roc_curve(auc_list, tpr_list, plot_name)
        return [tpr_list, auc_list, val_index_list, features_per_fold]

    def _cross_validation_single_fold(
            self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame,
            tpr_list: list, auc_list: list, features_per_fold: list,
            plot_name: str, **kwargs
    ) -> None:
        train_return = self._cross_validation_single_fold_train_and_add_basic_metrics(
            df_train_fold, df_val_fold, plot_name, tpr_list, auc_list, features_per_fold
        )

    # ---- Training ----
    def train(self, df_train: pd.DataFrame = None, df_other: pd.DataFrame = None,
              plot_name: str = 'attention_adjuvant_treatment', model_reset: bool = True,
              batch_size: int = 32, epochs: int = 10, **kwargs
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

             batch_size (int, optional): The batch size for the training process.
             Defaults to 32.

             epochs (int, optional): The number of epochs for the training process.

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
        # self._training_calculate_metrics expects the predicted probabilities
        # for class 1 only.
        scores = self._training_calculate_metrics(y_other[:, 1], y_pred)

        if self._plotter.plot_flag or self._plotter.save_flag:
            self._plot_train(df_other, y_pred, plot_name)
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

        for index, (ground_truth, prediction) in enumerate(zip(y_other[:, 1], y_pred)):
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
            colormap='coolwarm', fig_size=(12, 8), plot_name=plot_name
        )

    # ---- Prediction -----
    def predict(self, data: pd.DataFrame | np.ndarray) -> np.array:
        y_pred = self.model.predict(data)[:, 1]
        return y_pred

    # ---- Data Preparation ----
    def _prepare_data(self) -> None:
        raise NotImplementedError('Data preparation not implemented')

    def _prepare_data_split(self) -> None:
        data_split_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.data_split_treatment_outcome,
            data_dir=self.args.data_split_dir /
                     self._default_file_names.data_split_treatment_outcome,
            data_dir_flag=True
        )
        self._data_split = data_split_reader.return_data()
        self._data_split["target"] = self._data_split["adjuvant_treatment"].apply(
            lambda x: 0 if x == "none" else 1)

    # ---- Model Creation ----
    def _create_new_model(self) -> tf.keras.Model:
        raise NotImplementedError('Model creation not implemented')

    # ---- General useful methods ----
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


class AbstractAttentionMlpAdjuvantTreatmentPredictor(AbstractNeuralNetworkAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments with a multi layer perceptron
    where the features are fused with a simple attention mechanism.
    This is an abstract class and will throw a NotImplementedError if it is
    initialized directly.

    To implement:
        _prepare_data: Prepare the data for the df_tran and df_test properties.
        Use the _data_reader_factory to get the data and set it to the
        self._data attribute of the class.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    def _create_new_model(self) -> tf.keras.Model:
        learning_rate = 0.001
        model = AttentionMLPModel(self._input_dim)
        model = model.build_model(self._input_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model


class TmaTabularMergedAttentionMlpAdjuvantTreatmentPredictor(
    AbstractAttentionMlpAdjuvantTreatmentPredictor
):
    """Class for predicting adjuvant treatments with a multi layer perceptron
    where the features are fused with a simple attention mechanism.
    This implementation uses the merged tabular and tma feature data.

    Methods:
        cross_validate: Performs cross-validation on the training data.
        train: Trains the model on the training data and validates on the test data.
        predict: Predicts the adjuvant treatment for the given data.

    Properties:
        df_train: The training data.
        df_test: The test data.
        model: The model that is used for training and prediction
    """

    def _prepare_data(self) -> None:
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tma_tabular_merged_feature,
            data_dir=self.args.features_dir,
            data_dir_flag=True
        )
        self._data = data_reader.return_data()

    def prepare_data_for_model(
            self, df_train_fold: pd.DataFrame, df_other_fold: pd.DataFrame
    ) -> list:
        [[x_train, y_train, x_other, y_other], features] = super().prepare_data_for_model(df_train_fold, df_other_fold)

        feature_names_filtered = [value
                                  for value in features
                                  if value not in TMA_VECTOR_FEATURES]
        deleted_tma_features = [value
                                for value in features
                                if value not in feature_names_filtered]
        feature_tma_names = [f'{i}_{marker}' for i in range(0, TMA_VECTOR_LENGTH)
                             for marker in deleted_tma_features]
        features = np.concatenate((feature_names_filtered, feature_tma_names))

        return [[x_train, y_train, x_other, y_other], features]


class TabularMergedAdjuvantTreatmentPredictor(AdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the merged tabular
    data is used for the prediction.
    """
    def _prepare_data(self) -> None:
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tabular_merged_feature,
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

    def _create_new_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=600, min_samples_split=5,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features='sqrt', max_depth=50,
            criterion='log_loss',
            random_state=np.random.RandomState(self._random_number)
        )
        return model


class ClinicalAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the clinical tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.clinical_feature
        )
        self._data = data_reader.return_data()


class PathologicalAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the pathological tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.patho_feature
        )
        self._data = data_reader.return_data()


class BloodAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the blood tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.blood_feature
        )
        self._data = data_reader.return_data()


class TMACellDensityAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the TMA cell density tabular
    data is used.
    """

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.tma_cell_density_feature
        )
        self._data = data_reader.return_data()


class ICDCodesAdjuvantTreatmentPredictor(TabularMergedAdjuvantTreatmentPredictor):
    """Class for predicting adjuvant treatments. Here the ICD codes are
    is used."""

    def _prepare_data(self):
        data_reader = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.icd_codes_feature
        )
        self._data = data_reader.return_data()
