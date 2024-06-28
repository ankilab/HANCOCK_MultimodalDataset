from pathlib import Path


class DefaultPaths:
    def __init__(self):
        # --- Base paths
        self.results = Path(__file__).parents[1] / 'results'
        self.features = Path(__file__).parents[1] / 'features'
        self.data_dir = Path(__file__).parents[2] / 'Hancock_Dataset'
        self.structured_data_dir = self.data_dir / 'StructuredData'
        self.data_split = self.data_dir / 'DataSplits_DataDictionaries'

        # -- data paths
        self.clinical = self.structured_data_dir / "clinical_data.json"
        self.patho = self.structured_data_dir / "pathological_data.json"
        self.blood = self.structured_data_dir / "blood_data.json"
        self.blood_ref = self.structured_data_dir / \
            "blood_data_reference_ranges.json"
        self.celldensity = self.data_dir / \
            "TMA_CellDensityMeasurements" / "TMA_celldensity_measurements.csv"
        self.text_data_dir = self.data_dir / 'TextData'
        self.reports = self.text_data_dir / "reports"
        self.icd_codes = self.text_data_dir / 'icd_codes'
        self.wsi_tumor = self.data_dir / 'WSI_PrimaryTumor'
        self.wsi_lymph_node = self.data_dir / 'WSI_LymphNode'

        # --- Features -----
        self.feature_clinical_file_name = 'clinical.csv'
        self.feature_patho_file_name = 'pathological.csv'
        self.feature_blood_file_name = 'blood.csv'
        self.feature_icd_codes_file_name = 'icd_codes.csv'
        self.feature_cell_density_file_name = 'tma_cell_density.csv'

        self.feature_clinical = self.features / self.feature_clinical_file_name
        self.feature_patho = self.features / self.feature_patho_file_name
        self.feature_blood = self.features / self.feature_blood_file_name
        self.feature_icd_codes = self.features / self.feature_icd_codes_file_name
        self.feature_celldensity = self.features / self.feature_cell_density_file_name

        # ---- Targets ----
        self.targets_file_name = 'targets.csv'
        self.targets = self.features / self.targets_file_name

        # ---- Data Splits ----
        self.data_split_blood_file_name = 'DataDictionary_blood.csv'
        self.data_split_clinical_file_name = 'DataDictionary_clinical.csv'
        self.data_split_patho_file_name = 'DataDictionary_pathological.csv'
        self.data_split_in_file_name = 'dataset_split_in.json'
        self.data_split_orypharynx_file_name = 'dataset_split_Oropharynx.json'
        self.data_split_out_file_name = 'dataset_split_out.json'
        self.data_split_treatment_outcome_file_name = 'dataset_split_treatment_outcome.json'

        self.data_split_blood = self.data_split / self.data_split_blood_file_name
        self.data_split_clinical = self.data_split / self.data_split_clinical_file_name
        self.data_split_patho = self.data_split / self.data_split_patho_file_name
        self.data_split_in = self.data_split / self.data_split_in_file_name
        self.data_split_orypharynx = self.data_dir / \
            self.data_split_orypharynx_file_name
        self.data_split_out = self.data_dir / self.data_dir_out_file_name
        self.data_split_treatment_outcome = self.data_dir / \
            self.data_dir_treatment_outcome_file_name
