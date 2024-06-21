from data_reader import DataFrameReaderFactory
import os
import re
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


def get_file_count(file_dir, slide_type, subdirs=False):
    if subdirs:
        file_list = []
        for subdir in os.listdir(file_dir):
            if os.path.isdir(file_dir/subdir):
                for f in os.listdir(file_dir/subdir):
                    patient_id = re.search(r"[0-9]{3}", f).group()
                    file_list.append({
                        "patient_id": patient_id,
                        "file": file_dir/subdir/f
                    })
    else:
        file_list = []
        for f in os.listdir(file_dir):
            if not os.path.isdir(file_dir/f):
                patient_id = re.search(r"[0-9]{3}", f).group()
                file_list.append({
                    "patient_id": patient_id,
                    "file": file_dir / f
                })
    if (file_list == []):
        slide_df = pd.DataFrame(columns=["patient_id", "file"])
    else:
        slide_df = pd.DataFrame(file_list)
    count = slide_df["patient_id"].value_counts().reset_index()
    count.columns = ["patient_id", slide_type]
    return count


def load_measurements(measurement_file, tma_name):
    data = pd.read_csv(measurement_file, dtype={"Case ID": str})

    # drop missing cores
    missing_idx = data[data["Missing"] == True].index
    data = data.drop(missing_idx)

    # drop cores with no Case ID
    nan_idx = data[data["Case ID"].isna()].index
    data = data.drop(nan_idx)

    # Extract location
    data["location"] = data["Image"].str.extract(
        r"(TumorCenter|InvasionFront)")

    tma_z = data[data["location"] == "TumorCenter"]
    tma_inv = data[data["location"] == "InvasionFront"]

    count_z = tma_z["Case ID"].value_counts().reset_index()
    count_inv = tma_inv["Case ID"].value_counts().reset_index()

    count_z.columns = ["patient_id", tma_name + " tumor center"]
    count_inv.columns = ["patient_id", tma_name + " invasion front"]
    return count_z, count_inv


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

    def __init__(self, parser: ArgumentParser):
        self._add_parser_args(parser)
        self._create_absolute_paths(parser)

        dataFrameReaderFactory = DataFrameReaderFactory()
        self._clinicalDataFrameReader = dataFrameReaderFactory.make_data_frame_reader(
            'Clinical', self._clinical_path)
        self._pathoDataFrameReader = dataFrameReaderFactory.make_data_frame_reader(
            'Pathological', self._patho_path)
        self._bloodDataFrameReader = dataFrameReaderFactory.make_data_frame_reader(
            'Blood', self._blood_path)

        self._wsiPrimaryTumorDataFrameReader = dataFrameReaderFactory.make_data_frame_reader(
            'WSI_PrimaryTumor', self._prim_path
        )
        self._wSILymphNodeDataFrameReader = dataFrameReaderFactory.make_data_frame_reader(
            'WSI_LymphNode', self._lk_path
        )
        self._tmaCellDensityDataFrameReader = dataFrameReaderFactory.make_data_frame_reader(
            'TMA_CellDensityMeasurement', self._cell_density_path
        )
        

        self._merged = None

        rcParams.update({"font.size": 6})
        rcParams["svg.fonttype"] = "none"

    def _add_parser_args(self, parser: ArgumentParser) -> None:
        """Functions adds the available arguments to the parser. The arguments

        Args:
            parser (ArgumentParser): The parser to which the arguments are added.
        """
        parser.add_argument("--dataset_dir", type=str,
                            help="Root directory of dataset", nargs="?",
                            default=Path(__file__).parents[2] / 'Hancock_Dataset')
        parser.add_argument("--results_dir", type=str,
                            help="Directory where plot will be saved", nargs="?",
                            default=Path(__file__).parents[1] / 'results')
        parser.add_argument(
            "--path_clinical",
            type=str,
            help="Relative path of clinical data file in the specified dataset_dir",
            default="StructuredData/clinical_data.json",
            nargs="?"
        )
        parser.add_argument(
            "--path_patho",
            type=str,
            help="Relative path of pathological data file in the specified dataset_dir",
            default="StructuredData/pathological_data.json",
            nargs="?"
        )
        parser.add_argument(
            "--path_blood",
            type=str,
            help="Relative path of blood data file in the specified dataset_dir",
            default="StructuredData/blood_data.json",
            nargs="?"
        )
        parser.add_argument(
            "--dir_wsi_primarytumor",
            type=str,
            help="Relative path to the WSI_PrimaryTumor directory in the specified dataset_dir",
            default="WSI_PrimaryTumor",
            nargs="?"
        )
        parser.add_argument(
            "--dir_wsi_lymphnode",
            type=str,
            help="Relative path to the WSI_LymphNode directory in the specified dataset_dir",
            default="WSI_LymphNode",
            nargs="?"
        )
        parser.add_argument(
            "--path_celldensity",
            type=str,
            help="Relative path to the cell density measurements file in the specified dataset_dir",
            default="TMA_CellDensityMeasurements/TMA_celldensity_measurements.csv",
            nargs="?"
        )
        parser.add_argument(
            "--path_reports",
            type=str,
            help="Relative path to the directory containing surgery reports in the specified dataset_dir",
            default="TextData/reports/",
            nargs="?"
        )

    def _create_absolute_paths(self, parser: ArgumentParser) -> None:
        """Creates the absolute paths from the relative paths specified in 
        the parser.

        Args:
            parser (ArgumentParser): The parser with the relative paths 
            as arguments.
        """
        args = parser.parse_args()
        root_dir = args.dataset_dir
        self._clinical_path = root_dir / args.path_clinical
        self._patho_path = root_dir / args.path_patho
        self._blood_path = root_dir / args.path_blood
        self._prim_path = root_dir / args.dir_wsi_primarytumor
        self._lk_path = root_dir / args.dir_wsi_lymphnode
        self._cell_density_path = root_dir / args.path_celldensity
        self._report_path = root_dir / args.path_reports
        self._result_path = args.results_dir

    def get_tabular_data_count(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns the tabular data from the structured data files. 
        Reads them from the directory specified in the parser and retains 
        only two columns. The first column is the patient_id and the second 
        the number of available records for that patient. 

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The tabular data
            as dataframes for clinical, pathological and blood data.
        """
        clinical_count = self._clinicalDataFrameReader.return_data_count()
        patho_count = self._pathoDataFrameReader.return_data_count()
        blood_count = self._bloodDataFrameReader.return_data_count()

        return clinical_count, patho_count, blood_count

    def get_available_data_frame(self) -> pd.DataFrame:
        if self._merged is None:
            pass
        return self._merged.copy()


if __name__ == '__main__':
    parser = ArgumentParser()
    plotter = HancockAvailableDataPlotter(parser)
    args = parser.parse_args()
    root_dir = Path(args.dataset_dir)

    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"

    clinical_path = root_dir/args.path_clinical
    patho_path = root_dir/args.path_patho
    blood_path = root_dir/args.path_blood

    # Structured data
    [clinical_count, patho_count, blood_count] = plotter.get_tabular_data_count()

    prim_count = get_file_count(
        # Image data
        root_dir/args.dir_wsi_primarytumor, "WSI Primary tumor", subdirs=True)
    lk_count = get_file_count(
        root_dir/args.dir_wsi_lymphnode, "WSI Lymph node")

    # TMA data (only CD3 considered, as example)
    tma_cd3_z, tma_cd3_inv = load_measurements(
        measurement_file=root_dir/args.path_celldensity, tma_name="TMA CD3")

    # Text data
    report = get_file_count(root_dir/args.path_reports, "Surgery report")

    # Merge
    merged = clinical_count
    for df in [patho_count, blood_count, report, tma_cd3_z, tma_cd3_inv, prim_count, lk_count]:
        merged = merged.merge(df, on="patient_id", how="outer")

    merged = merged.sort_values(by="patient_id").reset_index(drop=True)
    merged[merged.columns[1:]] = (merged[merged.columns[1:]] >= 1).astype(int)

    # Lymph node slides are only available for patients with positive lymph nodes
    # merged = merged.merge(
    #     patho[["patient_id", "number_of_positive_lymph_nodes"]].fillna(0), on="patient_id")
    # merged["WSI Lymph node"] = merged.apply(
    #     lambda x: 2 if (x["number_of_positive_lymph_nodes"] == 0) & (
    #         x["WSI Lymph node"] == 0) else x["WSI Lymph node"],
    #     axis=1)
    # merged = merged.drop("number_of_positive_lymph_nodes", axis=1)

    # # Plot
    # merged_plot = merged[merged.columns[1:]]
    # avail_sorted = merged_plot.sort_values(
    #     by=list(reversed(merged_plot.columns)), ascending=True)
    # avail_sorted = avail_sorted.T

    # plt.figure(figsize=(6, 2))
    # ax = sns.heatmap(avail_sorted, cmap=[sns.color_palette("Dark2")[
    #     1], sns.color_palette("Set2")[0], (1, 1, 1)], cbar=False)
    # plt.xticks([1, 100, 200, 300, 400, 500, 600, 700, 763])
    # ax.set_xticklabels([1, 100, 200, 300, 400, 500, 600, 700, 763])
    # plt.tight_layout()
    # ax.hlines(list(range(0, 11)), *ax.get_xlim(), colors="white")

    # myColors = [sns.color_palette("Set2")[0], sns.color_palette("Dark2")[
    #     1], (1, 1, 1)]
    # cmap = LinearSegmentedColormap.from_list("Custom", myColors, len(myColors))

    # legend_handles = [Patch(facecolor=sns.color_palette("Set2")[0], label="Available"),
    #                   Patch(facecolor=sns.color_palette("Dark2")[1], label="Not available")]
    # plt.legend(handles=legend_handles, loc='center left',
    #            bbox_to_anchor=(1, 0.5), frameon=False)
    # sns.despine(bottom=False, left=True, offset=5)
    # plt.tight_layout()
    # plt.savefig(Path(args.results_dir) /
    #             "available_data.svg", bbox_inches="tight")
    # plt.savefig(Path(args.results_dir)/"available_data.png",
    #             bbox_inches="tight", dpi=300)
    # plt.show()

    # # Find patients that don't cover all modalities
    # patients_missing_data = merged[
    #     # & (all_counts['HE Slides LYM'] == 0))
    #     (merged["WSI Primary tumor"] == 0)
    #     | (merged["Clinical data"] == 0)
    #     | (merged["Pathological data"] == 0)
    #     | (merged["Blood data"] == 0)
    #     | (merged["Surgery report"] == 0)
    # ]
    # print(f'Patients with missing data: {len(patients_missing_data)}\n')
    # print(patients_missing_data)
