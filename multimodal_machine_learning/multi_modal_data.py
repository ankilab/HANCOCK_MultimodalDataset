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
from defaults import DefaultFileNames
from argument_parser import HancockArgumentParser
from data_reader import DataFrameReaderFactory
from data_exploration.umap_embedding import setup_preprocessing_pipeline


def cross_validation(dataframe, random_state, k, plot_name=None):
    """

    Parameters
    ----------
    dataframe : Pandas dataframe

    random_state :

    k :

    plot_name :

    Returns
    -------

    """

    tpr_list = []
    auc_list = []
    x_linspace = np.linspace(0, 1, 100)
    shap_values = []
    val_index_list = []
    features_per_fold = []

    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for train_idx, val_idx in cv.split(dataframe, dataframe["target"]):
        # Split folds
        df_train_folds = dataframe.iloc[train_idx]
        df_val_fold = dataframe.iloc[val_idx]

        # Preprocess data
        preprocessor = setup_preprocessing_pipeline(
            df_train_folds.columns[2:], min_max_scaler=True)

        x_train = preprocessor.fit_transform(
            df_train_folds.drop(["patient_id", "target"], axis=1))
        x_val = preprocessor.transform(
            df_val_fold.drop(["patient_id", "target"], axis=1))
        y_train = df_train_folds["target"].to_numpy()
        y_val = df_val_fold["target"].to_numpy()

        # Handle class imbalance
        smote = SMOTE(random_state=random_state)
        x_train, y_train = smote.fit_resample(x_train, y_train)

        # Fit ML model
        model = RandomForestClassifier(
            n_estimators=800, random_state=random_state)
        model.fit(x_train, y_train)

        # Get predictions for validation fold
        y_pred = model.predict_proba(x_val)[:, 1]

        # Get SHAP values
        svs = shap.TreeExplainer(model).shap_values(x_val)
        shap_values.append(svs[:, :, 1])  # class = 1 (adjuvant treatment yes)
        val_index_list.append(val_idx)
        features_per_fold.append(preprocessor.get_feature_names_out())

        # ROC curve
        fpr, tpr, thresh = roc_curve(y_val, y_pred)
        tpr = np.interp(x_linspace, fpr, tpr)
        tpr[0] = 0.0
        tpr[-1] = 1.0
        tpr_list.append(tpr)
        auc_list.append(roc_auc_score(y_val, y_pred))

    # SHAP
    # shap values: shape (#folds * #classes, #samples, #features)
    preprocessor = setup_preprocessing_pipeline(
        dataframe.columns[2:], min_max_scaler=True)
    preprocessor.set_output(transform="pandas")
    dataframe_reindex = preprocessor.fit_transform(
        dataframe.drop(["patient_id", "target"], axis=1))
    feature_names = preprocessor.get_feature_names_out()

    # Workaround: OneHotEncoder sometimes looses a category in k-fold cv
    for i in range(len(shap_values)):
        features = features_per_fold[i]
        missing = set(feature_names) - set(features)
        for m in missing:
            # Find index of missing category
            missing_idx = list(feature_names).index(m)
            # Insert zeros for missing category
            shap_values[i] = np.insert(shap_values[i], missing_idx, 0, axis=1)

    all_val_folds_idx = [
        idx for idx_fold in val_index_list for idx in idx_fold]
    shap.summary_plot(
        shap_values=np.concatenate(shap_values),
        features=dataframe_reindex.reindex(all_val_folds_idx),
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
    if plot_name is not None:
        plt.savefig(
            results_dir/f"shap_summary_plot_{plot_name}_test.svg", bbox_inches="tight")
        plt.savefig(
            results_dir/f"shap_summary_plot_{plot_name}_test.png", bbox_inches="tight", dpi=200)
    plt.close()

    return tpr_list, auc_list, shap_values


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


class HancockPredictor:
    """Abstract class that defines the interface for all predictors in the Hancock
    project. All predictors should inherit from this class.
    """
    @property
    def df_train(self) -> pd.DataFrame:
        if self._df_train is None:
            self._df_train = self._get_df_train()
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        if self._df_test is None:
            self._df_test = self._get_df_test()
        return self._df_test.copy()

    @property
    def model(self):
        if self._model is None:
            raise NotImplementedError("Model not trained yet.")
        return self._model.copy()

    def __init__(
        self, save_flag: bool = False, plot_flag: bool = False,
        random_state: int = 42, predictor_type: str = 'None'
    ):
        rcParams.update({"font.size": 6})
        rcParams["svg.fonttype"] = "none"
        self.argumentParser = HancockArgumentParser(
            type=predictor_type)
        self.args = self.argumentParser.parse_args()
        self.save_flag = save_flag
        self.plot_flag = plot_flag
        self.random_state = np.random.RandomState(random_state)
        self._data_reader_factory = DataFrameReaderFactory()

        self._df_train = None
        self._df_test = None
        self._model = None
        pass

    def cross_validate(
        self, n_splits: int = 10, plot_flag: bool = None, plot_name: str = 'cross_validate',
        save_flag: bool = None
    ) -> None:
        raise NotImplementedError("Cross-validation not implemented.")

    def train(self, plot_flag: bool = None, plot_name: str = 'train',
              save_flag: bool = None
              ) -> None:
        raise NotImplementedError("Training not implemented.")

    def test(self, plot_flag: bool = None, plot_name: str = 'test',
             save_flag: bool = None
             ) -> None:
        raise NotImplementedError("Testing not implemented.")

    def predict(self, data: pd.DataFrame) -> list:
        raise NotImplementedError("Prediction not implemented.")

    def _get_df_train(self) -> pd.DataFrame:
        raise NotImplementedError("Data frame for training not implemented.")

    def _get_df_test(self) -> pd.DataFrame:
        raise NotImplementedError("Data frame for testing not implemented.")


class AdjuvantTreatmentPredictor(HancockPredictor):
    def __init__(
        self, save_flag: bool = False, plot_flag: bool = False,
        random_state: int = 42
    ):
        super().__init__(save_flag=save_flag, plot_flag=plot_flag,
                         random_state=random_state,
                         predictor_type='adjuvant_treatment_prediction'
                         )
        self._prepare_data()
        self._prepare_data_split()

    def cross_validate(
        self, n_splits: int = 10, plot_flag: bool = None, plot_name: str = 'cross_validate',
        save_flag: bool = None
    ) -> None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in cv.split(self.df_train, self.df_train['target']):
            df_train_fold = self.df_train.iloc[train_idx]
            df_val_fold = self.df_train.iloc[val_idx]
            [x_train, y_train, x_val, y_val] = self._prepare_data_for_training(df_train_fold, df_val_fold)
            # ToDo: Continue adapting from cross_validation function

    def _prepare_data(self) -> None:
        factory = DataFrameReaderFactory()
        data_reader = factory.make_data_frame_reader(
            data_type=factory.data_reader_types.structural_aggregated,
            data_dir=self.args.features_dir,
            data_dir_flag=True
        )
        self._data = data_reader.return_data()
    
    def _prepare_data_split(self) -> None:
        factory = DataFrameReaderFactory()
        default_file_names = DefaultFileNames()
        data_split_reader = factory.make_data_frame_reader(
            data_type=factory.data_reader_types.data_split_treatment_outcome,
            data_dir=self.args.data_split_dir /
            default_file_names.data_split_treatment_outcome,
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

    def _prepare_data_for_training(self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame) -> list:
        """Preprocesses the input data and creates a numpy array for the labels. Also, over-samples the
        underrepresented class if necessary.

        Args:
            df_train_fold (pd.DataFrame): The data intended for training. Must have
            a column 'patient_id' and 'target'.
            df_val_fold (pd.DataFrame): The data intended for validating the training process.

        Returns:
            list: List with x_train, y_train, x_val and y_val in that order.
        """
        preprocessor = setup_preprocessing_pipeline(
            df_train_fold.columns[2:], min_max_scaler=True)

        x_train = preprocessor.fit_transform(
            df_train_fold.drop(["patient_id", "target"], axis=1))
        x_val = preprocessor.transform(
            df_val_fold.drop(["patient_id", "target"], axis=1))
        y_train = df_train_fold["target"].to_numpy()
        y_val = df_val_fold["target"].to_numpy()

        # Handle class imbalance
        smote = SMOTE(random_state=self.random_state)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        return [x_train, y_train, x_val, y_val]


if __name__ == "__main__":
    predictor = AdjuvantTreatmentPredictor()

    # data_split_dir = predictor.args.data_split_dir
    # results_dir = predictor.args.results_dir
    # features_dir = predictor.args.features_dir
    #
    # df_train = predictor.df_train
    # df_test = predictor.df_test

    # Train classifier on multimodal data with k-fold CV
    # print("Running k-fold cross-validation for multimodal data...")
    # roc_multimodal, auc_multimodal, shap_values_multimodal = cross_validation(
    #     df_train, predictor.random_state, k=2, plot_name="multimodal")

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
