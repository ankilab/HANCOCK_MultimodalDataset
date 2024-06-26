from pathlib import Path


class DefaultPaths:
    def __init__(self):
        self.results = Path(__file__).parents[1] / 'results'
        self.features = Path(__file__).parents[1] / 'features'
        self.data_dir = Path(__file__).parents[2] / 'Hancock_Dataset'
        self.structured_data_dir = self.data_dir / 'StructuredData'
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
        self.data_split = self.data_dir / 'DataSplits_DataDictionaries'
        # --- Features -----
        self.feature_clinical = self.features / 'clinical.csv'
        self.feature_patho = self.features / 'pathological.csv'
        self.feature_blood = self.features / 'blood.csv'
        self.feature_icd_codes = self.features / 'icd_codes.csv'
        self.feature_celldensity = self.features / 'tma_cell_density.csv'
        # ---- Targets ----
        self.targets = self.features / 'targets.csv'
        
    