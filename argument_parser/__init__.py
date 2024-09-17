from argparse import ArgumentParser
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parents[1]))
from defaults import DefaultPaths


class HancockArgumentParser(ArgumentParser):
    """Argument parser for the Hancock project. Sets always a 
    default option in case not all arguments are provided, or needed 
    for the specific use case. Takes in the initializer the type of 
    the script that calls to reduce the number of arguments available. 
    """
    def __init__(self, type: str = 'None', **kwargs):
        """Argument parser for the Hancock project. Sets always a 
        default option in case not all arguments are provided, or needed 
        for the specific use case. 

        Args:
            type (str, optional): The type of script calling. Should 
            be used to reduce the number of available arguments that 
            should not be used. Defaults to 'None'.
        """
        super().__init__(**kwargs)
        self._defaultPaths = DefaultPaths()
        self._add_always_arguments()
        if type == 'feature_extraction' or 'grading_correlation':
            self._add_structured_arguments()
            self._add_tma_cell_density_arguments()
            self._add_text_data_arguments()
            self._add_feature_extraction_arguments()
        elif type == 'plot_available_data':
            self._add_structured_arguments()
            self._add_tma_cell_density_arguments()
            self._add_text_data_arguments()
            self._add_wsi_arguments()
        elif type == 'adjuvant_treatment_prediction':
            self._add_data_split_arguments()
        elif type == 'adjuvant_treatment_prediction_convnet':
            self._add_data_split_arguments()
            self._add_convnet_arguments()

    def _add_always_arguments(self):
        self.add_argument("--results_dir", type=Path,
                          help="Directory where plot will be saved", nargs="?",
                          default=self._defaultPaths.results)
        self.add_argument(
            "--features_dir", type=Path, nargs='?',
            help="Path to directory where the features are saved",
            default=self._defaultPaths.features
        )

    def _add_structured_arguments(self):
        self.add_argument(
            "--path_clinical",
            type=Path,
            help="Absolute path of clinical data file.",
            default=self._defaultPaths.clinical,
            nargs="?"
        )
        self.add_argument(
            "--path_patho",
            type=Path,
            help="Absolute path of pathological data file.",
            default=self._defaultPaths.patho,
            nargs="?"
        )
        self.add_argument(
            "--path_blood",
            type=Path,
            help="Absolute path of blood data file.",
            default=self._defaultPaths.blood,
            nargs="?"
        )
        
        self.add_argument(
            "--path_blood_ref",
            type=Path,
            help="Absolute path of blood reference file.",
            default=self._defaultPaths.blood_ref,
            nargs="?"
        )

    def _add_tma_cell_density_arguments(self):
        self.add_argument(
            "--path_celldensity",
            type=Path,
            help="Absolute path to the cell density measurements file in the specified dataset_dir",
            default=self._defaultPaths.cell_density,
            nargs="?"
        )

    def _add_text_data_arguments(self):
        self.add_argument(
            "--path_reports",
            type=Path,
            help="Absolute path to the directory containing surgery reports in the specified dataset_dir",
            default=self._defaultPaths.reports,
            nargs="?"
        )
        self.add_argument(
            "--path_icd_codes", type=Path, nargs='?',
            help="Path to directory with ICD code text files",
            default=self._defaultPaths.icd_codes
        )

    def _add_wsi_arguments(self):
        self.add_argument(
            "--path_wsi_primarytumor",
            type=Path,
            help="Absolute path to the WSI_PrimaryTumor directory in the specified dataset_dir",
            default=self._defaultPaths.wsi_tumor,
            nargs="?"
        )
        self.add_argument(
            "--path_wsi_lymphnode",
            type=Path,
            help="Absolute path to the WSI_LymphNode directory in the specified dataset_dir",
            default=self._defaultPaths.wsi_lymph_node,
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
        
    def _add_data_split_arguments(self):
        self.add_argument(
            "--data_split_dir", type=Path, nargs='?',
            help="Path to directory that contains data splits as JSON files",
            default=self._defaultPaths.data_split
        )

    def _add_convnet_arguments(self):
        self.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
        self.add_argument("--prefix", dest="prefix", type=str, default="", help="Custom prefix for filenames")
