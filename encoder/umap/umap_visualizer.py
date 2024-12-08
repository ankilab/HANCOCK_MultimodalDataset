# ====================================================================================================================
# Imports
# ====================================================================================================================
import pandas as pd
import umap
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from encoder import (
    SurgeryReportEncoderPubMedBert, SurgeryReportEncoderBioClinicalBert,
    SurgeryReportEncoderTextEmbedding3Large,
    TabularMergedEncoder,
    Encoder,
    EncodingDataFrameCreator
)
from data_reader import DataFrameReaderFactory
from defaults import DefaultPaths


# ====================================================================================================================
# UMAP Visualizer
# ====================================================================================================================
class UmapVisualizer:
    @property
    def seed(self) -> int:
        return 42

    def __init__(
            self, encoder: Encoder, data_reader_factory: DataFrameReaderFactory = DataFrameReaderFactory(),
            visualizer_base_save_dir: Path = Path(__file__).parents[2] / 'results' / 'umap_visualizations' ,
            visualizer_base_name: str = 'umap_visualization_modality_encoder',
            clinical_data_path: Path = DefaultPaths().clinical,
            patho_data_path: Path = DefaultPaths().patho, umap_min_dist: float = 0.1, umap_n_neighbors: int =15
    ):
        self.encoder = encoder
        self.data_reader_factory = data_reader_factory
        self.visualizer_base_save_dir = visualizer_base_save_dir
        self.visualizer_base_name = visualizer_base_name
        self.encoding_data_frame_creator = EncodingDataFrameCreator(
            encoder=encoder, patho_data_path=patho_data_path, clinical_data_path=clinical_data_path
        )
        self.umap_min_dist = umap_min_dist
        self.umap_n_neighbors = umap_n_neighbors

        self._merged_data = None
        self._x_axis_label = 'UMAP 1'
        self._y_axis_label = 'UMAP 2'

    # ================================================================================================================
    # Public Methods
    # ================================================================================================================
    def visualize(self, save_flag: bool = True, force_new_encoding_creation: bool = False) -> None:
        """
        Visualize the encodings from the encoder using UMAP for different clinical and pathological attributes.

        Args:
            save_flag (bool, optional): Flag to save the visualizations. Defaults to True.
            force_new_encoding_creation (bool, optional): Flag to force the creation of new encodings. Defaults to False.
        """
        _, _ = self.return_umap_axis()
        subplot_titles, n_rows, n_cols = self._return_subplot_titles()
        fig, axes = self._create_plt_axes(n_rows, n_cols)

        for row, sublist in enumerate(subplot_titles):
            for col, feature in enumerate(sublist):
                if (feature in self._merged_data.columns or
                        feature == 'follow_up_months_deceased' or
                        feature == 'follow_up_months_living' or
                        feature == 'follow_up_months'
                ):
                    self._create_scatter_plot(axis=axes[row, col], feature=feature)
                else:
                    warnings.warn(f'Feature {feature} not found in the merged data and thus not visualized')

        sns.despine()
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, wspace=0.4)
        if save_flag:
            if not self.visualizer_base_save_dir.exists():
                self.visualizer_base_save_dir.mkdir(parents=True)
            plt.savefig(
                self.visualizer_base_save_dir / f'{self.visualizer_base_name}.svg', bbox_inches="tight",
                dpi=200
            )
        plt.show()

    def create_merged_data_frame(self) -> pd.DataFrame:
        if self._merged_data is not None:
            return self._merged_data.copy()

        self._merged_data = self.encoding_data_frame_creator.create_combined_data_frame()

        return self._merged_data.copy()

    def return_umap_axis(self) -> [np.ndarray, np.ndarray]:
        df = self.create_merged_data_frame()
        if self._x_axis_label in df.columns and self._y_axis_label in df.columns:
            return df[self._x_axis_label].values, df[self._y_axis_label].values

        umap_representation = umap.UMAP(random_state=self.seed, min_dist=self.umap_min_dist,
                                        n_neighbors=self.umap_n_neighbors).fit_transform(
            np.vstack(df['encoding'].values)
        )
        tx, ty = umap_representation[:, 0], umap_representation[:, 1]
        tx = (tx - min(tx)) / (max(tx) - min(tx))
        ty = (ty - min(ty)) / (max(ty) - min(ty))

        self._merged_data[self._x_axis_label] = tx
        self._merged_data[self._y_axis_label] = ty
        return tx, ty

    # ================================================================================================================
    # Helper Methods
    # ================================================================================================================
    def _create_scatter_plot(self, axis: any, feature: str) -> None:
        marker_size = 2
        if feature == 'follow_up_months_deceased':
            self._create_survival_months_scatter_plot(axis=axis, marker_size=marker_size, filter_value='deceased')
        elif feature == 'follow_up_months_living':
            self._create_survival_months_scatter_plot(axis=axis, marker_size=marker_size, filter_value='living')
        elif feature == 'follow_up_months':
            self._create_survival_months_scatter_plot(axis=axis, marker_size=marker_size, filter_value=None)
        elif self._feature_is_numerical(feature):
            self._create_complete_scatter_plot(axis=axis, feature=feature, marker_size=marker_size)
        else:
            self._create_categorical_scatter_plot(axis=axis, feature=feature, marker_size=marker_size)

        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        axis.set_title(feature.replace("_", " "), fontsize=6)

    def _create_complete_scatter_plot(self, axis: any, feature: str, marker_size: int = 2):
        df = self.create_merged_data_frame()
        self._create_numeric_scatter_plot(df=df, feature=feature, axis=axis, marker_size=marker_size)

    def _create_categorical_scatter_plot(self, axis: any, feature: str, marker_size: int = 2):
        df = self.create_merged_data_frame().fillna('missing')
        palette = sns.color_palette(
            'Set2', n_colors=len(df[feature].unique()))
        legend = True
        sns.scatterplot(
            df, x=self._x_axis_label, y=self._y_axis_label,
            hue=feature, hue_norm=None,
            palette=palette, legend=legend, s=marker_size, ax=axis
        )
        legend = axis.legend(
            bbox_to_anchor=(1.02, 1), frameon=False,
            loc="upper left", borderaxespad=0, fontsize=4
        )
        for handle in legend.legend_handles:
            handle.set_markersize(4)

    def _create_survival_months_scatter_plot(
            self, axis: any, marker_size: int = 2, filter_value: str = None
    ) -> None:
        feature = 'follow_months_deceased'
        x, y  = self.return_umap_axis()
        df = self.create_merged_data_frame()
        days_per_month = 365 / 12.0
        df[feature] = np.round(df['days_to_last_information'] / days_per_month)
        if filter_value:
            df_relevant = df[df['survival_status'] == filter_value]
        else:
            df_relevant = df
        self._create_numeric_scatter_plot(df_relevant, feature, axis, marker_size)

    def _create_numeric_scatter_plot(self, df: pd.DataFrame, feature: str, axis:any, marker_size: int = 2):
        [palette, sm, hue_norm] = self._create_hue_norm(min_value=df[feature].min().min(),
                                                        max_value=df[feature].max().max())
        sns.scatterplot(
            df, x=self._x_axis_label, y=self._y_axis_label,
            hue_norm=hue_norm, hue=feature, legend=None,
            palette=palette, s=marker_size, ax=axis
        )
        plt.colorbar(sm, ax=axis)

    @staticmethod
    def _create_hue_norm(color_palette: str = 'flare', min_value: float = 0, max_value: float = 1.0):
        palette = sns.color_palette('flare', as_cmap=True)
        sm = plt.cm.ScalarMappable(
            cmap=palette, norm=plt.Normalize(min_value, max_value))
        hue_norm = sm.norm
        return palette, sm, hue_norm

    def _feature_is_numerical(self, feature: str) -> bool:
        df = self.create_merged_data_frame()
        if feature in df.columns:
            non_na_values = df[feature].dropna()
            if pd.api.types.is_numeric_dtype(non_na_values):
                return True
        return False

    # ================================================================================================================
    # Static Methods
    # ================================================================================================================
    @staticmethod
    def _return_subplot_titles() -> [list[list[str], int, int]]:
        subplot_titles = [
            ['sex', 'primarily_metastasis', 'smoking_status', 'age_at_initial_diagnosis'],
            ['pT_stage', 'pN_stage', 'grading', 'number_of_positive_lymph_nodes'],
            ['perinodal_invasion', 'lymphovascular_invasion_L', 'vascular_invasion_V', 'perineural_invasion_Pn'],
            ['infiltration_depth_in_mm', 'histologic_type', 'hpv_association_p16', 'primary_tumor_site'],
            ['survival_status', 'follow_up_months', 'follow_up_months_deceased', 'follow_up_months_living']

        ]
        n_rows = len(subplot_titles)
        n_cols = 4
        return subplot_titles, n_rows, n_cols

    @staticmethod
    def _create_plt_axes(n_rows: int, n_cols: int) -> [plt.Figure, any]:
        rcParams.update({'font.size': 4})
        rcParams['svg.fonttype'] = 'none'
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(int(1.75 * n_cols), n_rows))
        return [fig, axes]


# ====================================================================================================================
# Execution
# ====================================================================================================================
if __name__ == '__main__':
    encoders = []
    visualizer_base_names = []
    data_reader_factory = DataFrameReaderFactory()
    default_paths = DefaultPaths()
    clinical_data_path = default_paths.clinical
    patho_data_path = default_paths.patho

    # encoders.append(SurgeryReportEncoderPubMedBert())
    # visualizer_base_names.append('surgery_text_pub_med_bert_umap_visualization')
    #
    # encoders.append(SurgeryReportEncoderBioClinicalBert())
    # visualizer_base_names.append('surgery_text_bio_clinical_bert_umap_visualization')
    #
    # encoders.append(SurgeryReportEncoderTextEmbedding3Large())
    # visualizer_base_names.append('surgery_text_embedding_3_large_umap_visualization')
    #
    # # Is different to Marion's approach because we use also the patients where we have no entries in
    # # feature columns
    encoders.append(TabularMergedEncoder())
    visualizer_base_names.append('tabular_merged_umap_visualization')


    for encoder, visualizer_base_name in zip(encoders, visualizer_base_names):
        umap_visualizer = UmapVisualizer(encoder=encoder, data_reader_factory=data_reader_factory,
                                            visualizer_base_name=visualizer_base_name,
                                         clinical_data_path=clinical_data_path, patho_data_path=patho_data_path)
        umap_visualizer.visualize(save_flag=True)
