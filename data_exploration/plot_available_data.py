import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib import rcParams
import seaborn as sns
from argparse import ArgumentParser
from pathlib import Path
import sys

# Add root directory to system path to import DataFrameReaderFactory
sys.path.append(str(Path(__file__).parents[1]))
from argument_parser import HancockArgumentParser
from data_reader import DataFrameReaderFactory


class HancockAvailableDataPlotter:
    """Plots the available data from the hancock dataset.

    Default directory structure is:
    - dataset_dir
        - StructuredData
            - clinical_data.json
            - pathological_data.json
            - blood_data.json
        - WSI_PrimaryTumor
        - WSI_LymphNode
        - TMA_CellDensityMeasurements
            - TMA_celldensity_measurements.csv
        - TextData
            - reports
    """
    @property
    def merged(self) -> pd.DataFrame:
        if self._merged is None:
            self._merged = self._get_available_data()
        return self._merged.copy()

    def __init__(self):
        self._argumentParser = HancockArgumentParser(
            type='plot_available_data')
        self._create_absolute_paths(self._argumentParser)
        self._create_data_reader()

        self._merged = None

        rcParams.update({"font.size": 6})
        rcParams["svg.fonttype"] = "none"

    def _create_data_reader(self):
        factory = DataFrameReaderFactory()
        self._clinicalDataFrameReader = factory.make_data_frame_reader(
            factory.data_reader_types.clinical, self._clinical_path, data_dir_flag=True
        )
        self._pathoDataFrameReader = factory.make_data_frame_reader(
            factory.data_reader_types.patho, self._patho_path, data_dir_flag=True
        )
        self._bloodDataFrameReader = factory.make_data_frame_reader(
            factory.data_reader_types.blood, self._blood_path, data_dir_flag=True
        )

        self._wsiPrimaryTumorDataFrameReader = factory.make_data_frame_reader(
            factory.data_reader_types.wsi_tumor, self._prim_path, data_dir_flag=True
        )
        self._wSILymphNodeDataFrameReader = factory.make_data_frame_reader(
            factory.data_reader_types.wsi_lymph_node, self._lk_path, data_dir_flag=True
        )
        self._tmaCellDensityDataFrameReader = factory.make_data_frame_reader(
            factory.data_reader_types.tma_cell_density, self._cell_density_path, data_dir_flag=True
        )
        self._textDataReportsDataFrameReader = factory.make_data_frame_reader(
            factory.data_reader_types.text_reports, self._report_path, data_dir_flag=True
        )

    def _create_absolute_paths(self, parser: ArgumentParser) -> None:
        """Creates the absolute paths from the relative paths specified in 
        the parser.

        Args:
            parser (ArgumentParser): The parser with the relative paths 
            as arguments.
        """
        args = parser.parse_args()
        self._clinical_path = args.path_clinical
        self._patho_path = args.path_patho
        self._blood_path = args.path_blood
        self._prim_path = args.path_wsi_primarytumor
        self._lk_path = args.path_wsi_lymphnode
        self._cell_density_path = args.path_celldensity
        self._report_path = args.path_reports
        self._result_path = args.results_dir

    def _get_available_data(self) -> pd.DataFrame:
        """Creates a data frame with the available data for each patient. The 
        values in the dataframe can be either 0 (not available) or 1 (available) 
        and 2 (not available but makes sense because of diagnosis).

        Returns:
            pd.DataFrame: The data frame with the available data for each patient.
        """
        merged = self._merge_data()
        merged.sort_values(by='patient_id').reset_index(
            drop=True, inplace=True)
        merged[merged.columns[1:]] = (
            merged[merged.columns[1:]] >= 1).astype(int)
        merged = self._modify_lymph_node_data(merged)

        return merged

    def _merge_data(self) -> pd.DataFrame:
        """Creates a data frame with the available data for each patient. The 
        values in the dataframe are the counts of records for each patient for 
        the in the column indicated 'modality'. 

        Returns:
            pd.DataFrame: The data frame with the available data for each patient.
        """
        df_list = []
        merged = self._clinicalDataFrameReader.return_data_count()
        df_list.append(self._pathoDataFrameReader.return_data_count())
        df_list.append(self._bloodDataFrameReader.return_data_count())
        df_list.append(
            self._textDataReportsDataFrameReader.return_data_count())
        df_list.append(self._tmaCellDensityDataFrameReader.return_data_count())
        df_list.append(
            self._wsiPrimaryTumorDataFrameReader.return_data_count())
        df_list.append(self._wSILymphNodeDataFrameReader.return_data_count())
        for df in df_list:
            merged = pd.merge(merged, df, on='patient_id', how='outer')

        merged.sort_values(by='patient_id').reset_index(
            drop=True, inplace=True)

        return merged

    def _modify_lymph_node_data(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Sets all WSI Lymph node values to 2 if the patient has no positive
        lymph nodes.

        Args:
            merged (pd.DataFrame): Data frame with column 'patient_id' and
            'WSI Lymph node'.

        Returns:
            pd.DataFrame: Copy of merged data frame with modified 'WSI Lymph node' 
            column.
        """
        patho = self._pathoDataFrameReader.return_data()
        merged_copy = pd.merge(merged, patho[[
                               'patient_id', 'number_of_positive_lymph_nodes']]
                               .fillna(0), on='patient_id')
        merged_copy['WSI Lymph node'] = merged_copy.apply(
            lambda x: 2 if (x['number_of_positive_lymph_nodes'] == 0) & (
                x['WSI Lymph node'] == 0) else x['WSI Lymph node'], axis=1
        )
        merged_copy.drop('number_of_positive_lymph_nodes',
                         axis=1, inplace=True)

        return merged_copy

    def plot_available_data(self) -> None:
        """Plots the available data for each patient in a horizontal bar 
        chart. Y-axis does not correspond to the patient-id but is before
        sorted by the reversed list of columns. 
        """
        color_available = sns.color_palette('Set2')[0]
        color_not_available = sns.color_palette('Dark2')[1]
        color_not_available_lymph = sns.color_palette('Grays')[1]

        merged = self.merged
        merged_plot = merged[merged.columns[1:]]

        avail_sorted = merged_plot.sort_values(
            by=list(reversed(merged_plot.columns)), ascending=True)
        avail_sorted = avail_sorted.T

        plt.figure(figsize=(6, 2))
        ax = sns.heatmap(
            avail_sorted, cbar=False,
            cmap=[color_not_available, color_available, color_not_available_lymph]
        )
        plt.xticks([1, 100, 200, 300, 400, 500, 600, 700, 763])
        ax.set_xticklabels([1, 100, 200, 300, 400, 500, 600, 700, 763])
        plt.tight_layout()
        ax.hlines(list(range(0, 11)), *ax.get_xlim(), colors="white")

        legend_handles = [Patch(facecolor=color_available, label="Available"),
                          Patch(facecolor=color_not_available, label="Not available (N/A.)"),
                          Patch(facecolor=color_not_available_lymph, label="No lymph resection")]
        plt.legend(handles=legend_handles, loc='center left',
                   bbox_to_anchor=(1, 0.5), frameon=False)
        sns.despine(bottom=False, left=True, offset=5)
        plt.tight_layout()
        plt.savefig(self._result_path /
                    "available_data.svg", bbox_inches="tight")
        plt.savefig(self._result_path / "available_data.png",
                    bbox_inches="tight", dpi=300)
        plt.show()

    def print_missing_data(self) -> None:
        """Prints the rows in the available data where data is missing. 
        Ignores the WSI Primary tumor data as well as the WSI Lymph node data.
        For latter one this is performed because the data is not necessary for 
        each patient, depending on the diagnosis. 
        The WSI Primary tumor data is currently ignored because the 
        data is not available for any patient. 
        """
        patients_missing_data = self.merged[
            # & (all_counts['HE Slides LYM'] == 0))
            (self.merged["Clinical data"] == 0)
            | (self.merged["Pathological data"] == 0)
            | (self.merged["Blood data"] == 0)
            | (self.merged["Surgery report"] == 0)
            # | (self.merged["WSI Primary tumor"] == 0)
        ]
        print(f'Patients with missing data: {len(patients_missing_data)}\n')
        print(patients_missing_data)


if __name__ == '__main__':
    plotter = HancockAvailableDataPlotter()
    plotter.plot_available_data()
    plotter.print_missing_data()
