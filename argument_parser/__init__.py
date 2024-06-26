from argparse import ArgumentParser
from pathlib import Path

default_data_dir = Path(__file__).parents[2] / 'Hancock_Dataset'


class HancockArgumentParser(ArgumentParser):
    def __init__(self, type: str = 'None', **kwargs):
        super().__init__(**kwargs)
        self._add_always_arguments()
        if type == 'feature_extraction':
            self._add_structured_arguments()
            self._add_tma_cell_density_arguments()
            self._add_text_data_arguments()
            self._add_feature_extraction_arguments()
        elif type == 'plot_available_data':
            self._add_structured_arguments()
            self._add_tma_cell_density_arguments()
            self._add_text_data_arguments()
            self._add_wsi_arguments()

    def _add_always_arguments(self):
        self.add_argument("--results_dir", type=str,
                          help="Directory where plot will be saved", nargs="?",
                          default=Path(__file__).parents[1] / 'results')
        self.add_argument(
            "--features_dir", type=str, nargs='?',
            help="Path to directory where the features are saved",
            default=Path(__file__).parents[1] / 'features'
        )

    def _add_structured_arguments(self):
        default_structured_data_dir = default_data_dir / 'StructuredData'
        self.add_argument(
            "--path_clinical",
            type=str,
            help="Absolute path of clinical data file in the specified dataset_dir",
            default=default_structured_data_dir / "clinical_data.json",
            nargs="?"
        )
        self.add_argument(
            "--path_patho",
            type=str,
            help="Absolute path of pathological data file in the specified dataset_dir",
            default=default_structured_data_dir / "pathological_data.json",
            nargs="?"
        )
        self.add_argument(
            "--path_blood",
            type=str,
            help="Relative path of blood data file in the specified dataset_dir",
            default=default_structured_data_dir / "blood_data.json",
            nargs="?"
        )

    def _add_tma_cell_density_arguments(self):
        self.add_argument(
            "--path_celldensity",
            type=str,
            help="Absolute path to the cell density measurements file in the specified dataset_dir",
            default=default_data_dir / "TMA_CellDensityMeasurements" /
            "TMA_celldensity_measurements.csv",
            nargs="?"
        )

    def _add_text_data_arguments(self):
        default_text_data_dir = default_data_dir / 'TextData'
        self.add_argument(
            "--path_reports",
            type=str,
            help="Absolute path to the directory containing surgery reports in the specified dataset_dir",
            default=default_text_data_dir / "reports",
            nargs="?"
        )
        self.add_argument(
            "--path_icd_codes", type=str, nargs='?',
            help="Path to directory with ICD code text files",
            default=default_text_data_dir / 'icd_codes'
        )

    def _add_wsi_arguments(self):
        self.add_argument(
            "--path_wsi_primarytumor",
            type=str,
            help="Absolute path to the WSI_PrimaryTumor directory in the specified dataset_dir",
            default=default_data_dir / "WSI_PrimaryTumor",
            nargs="?"
        )
        self.add_argument(
            "--path_wsi_lymphnode",
            type=str,
            help="Absolute path to the WSI_LymphNode directory in the specified dataset_dir",
            default=default_data_dir / "WSI_LymphNode",
            nargs="?"
        )

    def _add_feature_extraction_arguments(self):
        self.add_argument(
            "-v", "--verbose", action="store_true",
            help="Show additional information"
        )
        self.add_argument(
            "--npz", action="store_true",
            help="Save features to compressed numpy file instead of csv"
        )
