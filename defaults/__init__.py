from pathlib import Path


class DefaultFileNames:
    def __init__(self):
        # ---- Features ----
        self.feature_clinical = 'clinical.csv'
        self.feature_patho = 'pathological.csv'
        self.feature_blood = 'blood.csv'
        self.feature_icd_codes = 'icd_codes.csv'
        self.feature_cell_density = 'tma_cell_density.csv'
        self.feature_tma_cd3 = 'centertile_dtr_256dim_CD3.npz'
        self.feature_tma_cd8 = 'centertile_dtr_256dim_CD8.npz'
        self.feature_tma_cd56 = 'centertile_dtr_256dim_CD56.npz'
        self.feature_tma_cd68 = 'centertile_dtr_256dim_CD68.npz'
        self.feature_tma_cd163 = 'centertile_dtr_256dim_CD163.npz'
        self.feature_tma_he = 'centertile_dtr_256dim_HE.npz'
        self.feature_tma_mhc1 = 'centertile_dtr_256dim_MHC1.npz'
        self.feature_tma_pdl1 = 'centertile_dtr_256dim_PDL1.npz'

        # ---- Targets ----
        self.targets = 'targets.csv'

        # ---- Data Splits ----
        self.data_split_blood = 'DataDictionary_blood.csv'
        self.data_split_clinical = 'DataDictionary_clinical.csv'
        self.data_split_patho = 'DataDictionary_pathological.csv'
        self.data_split_in = 'dataset_split_in.json'
        self.data_split_oropharynx = 'dataset_split_Oropharynx.json'
        self.data_split_out = 'dataset_split_out.json'
        self.data_split_treatment_outcome = 'dataset_split_treatment_outcome.json'


class DefaultPaths:
    def __init__(self):
        file_names = DefaultFileNames()
        
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
        self.wsi_tumor = self.data_dir / '..' / 'WSI_PrimaryTumor'
        self.wsi_lymph_node = self.data_dir / 'WSI_LymphNode'

        # --- Features -----
        self.feature_clinical = self.features / file_names.feature_clinical
        self.feature_patho = self.features / file_names.feature_patho
        self.feature_blood = self.features / file_names.feature_blood
        self.feature_icd_codes = self.features / file_names.feature_icd_codes
        self.feature_cell_density = self.features / file_names.feature_cell_density
        self.feature_tma_cd3 = self.features / file_names.feature_tma_cd3
        self.feature_tma_cd8 = self.features / file_names.feature_tma_cd8
        self.feature_tma_cd56 = self.features / file_names.feature_tma_cd56
        self.feature_tma_cd68 = self.features / file_names.feature_tma_cd68
        self.feature_tma_cd163 = self.features / file_names.feature_tma_cd163
        self.feature_tma_he = self.features / file_names.feature_tma_he
        self.feature_tma_mhc1 = self.features / file_names.feature_tma_mhc1
        self.feature_tma_pdl1 = self.features / file_names.feature_tma_pdl1

        # ---- Targets ----
        self.targets = self.features / file_names.targets

        # ---- Data Splits ----
        self.data_split_blood = self.data_split / file_names.data_split_blood
        self.data_split_clinical = self.data_split / file_names.data_split_clinical
        self.data_split_patho = self.data_split / file_names.data_split_patho
        self.data_split_in = self.data_split / file_names.data_split_in
        self.data_split_oropharynx = self.data_split / \
                                     file_names.data_split_oropharynx
        self.data_split_out = self.data_split / file_names.data_split_out
        self.data_split_treatment_outcome = self.data_split / \
            file_names.data_split_treatment_outcome


    