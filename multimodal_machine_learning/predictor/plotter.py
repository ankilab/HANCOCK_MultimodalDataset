# ===================================================================================================================
# Imports
# ====================================================================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, statistics
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import shap


# ===================================================================================================================
# Helper functions
# ====================================================================================================================
def get_significance(p_value) -> str:
    if p_value <= 0.001:
        return "$p\\leq$0.001 (***)"
    elif p_value <= 0.01:
        return "$p\\leq$0.01 (**)"
    elif p_value <= 0.05:
        return "$p\\leq$0.05 (*)"
    else:
        return f"$p=${p_value: .3f}"


# ===================================================================================================================
# Plotter Class
# ====================================================================================================================
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
        labels = ['Predicted False', 'Predicted True']
        if 'adjuvant_treatment' in plot_name or 'adjuvant_therapy' in plot_name:
            labels = ['Predicted no adjuvant therapy', 'Predicted adjuvant therapy']
        elif 'survival' in plot_name:
            labels = ['Predicted no survival', 'Predicted survival']
        elif 'recurrence' in plot_name:
            labels = ['Predicted no recurrence', 'Predicted recurrence']

        plt.figure(figsize=(3, 0.8))
        num_pred_adjuvant = np.count_nonzero(y_pred)
        num_pred_none = np.count_nonzero(y_pred == 0)
        ax = plt.barh([0, 1],
                      [num_pred_none, num_pred_adjuvant], color=sns.color_palette("Set2")[1])
        plt.bar_label(ax, fontsize=6, padding=2)
        plt.yticks(
            ticks=[0, 1],
            labels=labels
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
            self, shap_values: list, data_preprocessed: pd.DataFrame | list,
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
            self, shap_values: list, data_preprocessed: pd.DataFrame | list,
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
        it will be shown to the user, if the self.save_flag is True it will be saved to disk.

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
            colormap='coolwarm', fig_size=(3, 1.4), plot_name='lime_plot'
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

        # cmap = mcolors.LinearSegmentedColormap('my_cmap', shap.plots.colors.red_blue)
        # colors = [shap.plots.colors.red_blue(i / (256 - 1)) for i in range(256)]
        plt.figure(figsize=(3, 2.6))
        cmap = sns.color_palette(colormap, as_cmap=True)
        scatter = sns.scatterplot(
            data=df,
            x='Feature Contribution Value',
            y='Feature Name',
            hue='Feature Contribution Value',
            palette=cmap,
            s=15,  # Marker size
            alpha=0.7,
            legend=False
        )
        sm = plt.cm.ScalarMappable(cmap=cmap)
        cbar = plt.colorbar(sm, ax=plt.gca())

        plt.ylabel('Feature Name')
        plt.xlabel('Feature Contribution Value')
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.title(f'LIME Summary Plot - Top {n_features} Most Frequent Features')
        plt.tight_layout()

        if self.save_flag:
            plt.savefig(self.save_dir / f'{plot_name}.png')
            plt.savefig(self.save_dir / f'{plot_name}.svg')
        if self.plot_flag:
            plt.show()
        else:
            plt.close()

