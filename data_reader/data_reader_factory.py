from pathlib import Path
from .data_reader import (
    DataFrameReader,
    PathologicalDataFrameReader,
    ClinicalDataFrameReader,
    BloodDataFrameReader,
    WSIPrimaryTumorDataFrameReader,
    WSILymphNodeDataFrameReader,
    TMACellDensityDataFrameReader,
    TextDataReportsDataFrameReader,
    FeatureClinicalDataFrameReader,
    FeaturePathologicalDataFrameReader,
    FeatureBloodDataFrameReader,
    FeatureICDCodesDataFrameReader,
    FeatureTMACellDensityDataFrameReader,
    TargetsDataFrameReader,
    StructuralAggregatedDataFrameReader,
    DataSplitBloodDataFrameReader,
    DataSplitClinicalDataFrameReader,
    DataSplitPathologicalDataFrameReader,
    DataSplitInDataFrameReader,
    DataSplitOrypharynxDataFrameReader,
    DataSplitOutDataFrameReader,
    DataSplitTreatmentOutcomeDataFrameReader
)
import warnings


class DataReaderTypes:
    def __init__(self):
        # ---- Data ----
        self.patho = 'Pathological'
        self.clinical = 'Clinical'
        self.blood = 'Blood'
        self.structural_aggregated = 'Structural Aggregated'
        self.wsi_tumor = 'WSI_PrimaryTumor'
        self.wsi_lymph_node = 'WSI_LymphNode'
        self.tma_cell_density = 'TMA_CellDensityMeasurement'
        self.text_reports = 'TextData_reports'

        # ---- Features ----
        self.clinical_feature = 'Clinical Feature'
        self.patho_feature = 'Pathological Feature'
        self.blood_feature = 'Blood Feature'
        self.icd_codes_feature = 'ICD Codes Feature'
        self.tma_cell_density_feature = 'Feature TMA Cell Density'

        # ---- Targets ----
        self.targets = 'Targets'

        # ---- Data Split ----
        self.data_split_blood = 'Data Split Blood'
        self.data_split_clinical = 'Data Split Clinical'
        self.data_split_patho = 'Data Split Pathological'
        self.data_split_in = 'Data Split In'
        self.data_split_orypharynx = 'Data Split Orypharynx'
        self.data_split_out = 'Data Split Out'
        self.data_split_treatment_outcome = 'Data Split Treatment Outcome'


class DataFrameReaderFactory:
    """The DataReaderFactory class creates the appropriate DataReader object based on the 
    data type provided by the user.
    """

    def __init__(self):
        self.data_reader_types = DataReaderTypes()

    def make_data_frame_reader(
        self, data_type: str = 'NA', data_dir: Path = Path(__file__),
        data_dir_flag: bool = False
    ) -> DataFrameReader:
        """Returns the appropriate Data frame reader based on the data type 
        provided by the user.

        Args:
            data_type (str, optional): The kind of the data reader the user 
            want to receive. Defaults to 'NA'.

            data_dir (Path, optional): An optional path where the file to 
            be read is located. Defaults to Path(__file__).

            data_dir_flag (bool, optional): Only when this flag is 
            set to True, the inserted data_dir is considered. Otherwise 
            the default directory from the implemented data reader is used. 
            Defaults to False.

        Returns:
            DataFrameReader: DataFrameReader based on the given data_type.
        """
        data_reader = DataFrameReader
        data_reader = self._make_data_data_frame_reader(data_type, data_reader)
        data_reader = self._make_feature_data_frame_reader(
            data_type, data_reader)
        data_reader = self._make_data_split_data_frame_reader(
            data_type, data_reader)

        if data_reader == DataFrameReader:
            warnings.warn(f"Data type {data_type} not found. Returning default data reader.\n")
        
        if data_dir_flag:
            data_reader = data_reader(data_dir)
        else:
            data_reader = data_reader()
        
        return data_reader

    def _make_data_data_frame_reader(
        self, data_type: str, data_reader: DataFrameReader
    ) -> DataFrameReader:
        """Checks the data_type and returns the appropriate reference to the
        data reader class. If there is not match between the data_type and 
        the data related data_type's from the DataReaderTypes class, the
        input data_reader is returned.

        Args:
            data_type (str): Data type of the data reader that should be returned.
            data_reader (DataFrameReader): Reference to a data reader class.

        Returns:
            DataFrameReader: Reference to a data reader class based on the 
            data type.
        """
        if data_type == self.data_reader_types.patho:
            data_reader = PathologicalDataFrameReader
        elif data_type == self.data_reader_types.clinical:
            data_reader = ClinicalDataFrameReader
        elif data_type == self.data_reader_types.blood:
            data_reader = BloodDataFrameReader
        elif data_type == self.data_reader_types.wsi_tumor:
            data_reader = WSIPrimaryTumorDataFrameReader
        elif data_type == self.data_reader_types.wsi_lymph_node:
            data_reader = WSILymphNodeDataFrameReader
        elif data_type == self.data_reader_types.tma_cell_density:
            data_reader = TMACellDensityDataFrameReader
        elif data_type == self.data_reader_types.text_reports:
            data_reader = TextDataReportsDataFrameReader
        elif data_type == self.data_reader_types.structural_aggregated:
            data_reader = StructuralAggregatedDataFrameReader

        return data_reader

    def _make_feature_data_frame_reader(
        self, data_type: str, data_reader: DataFrameReader
    ) -> DataFrameReader:
        """Checks the data_type and returns the appropriate reference to the
        data reader class. If there is not match between the data_type and 
        the data related data_type's from the DataReaderTypes class, the
        input data_reader is returned.

        Args:
            data_type (str): Data type of the data reader that should be returned.
            data_reader (DataFrameReader): Reference to a data reader class.

        Returns:
            DataFrameReader: Reference to a data reader class based on the 
            data type.
        """
        if data_type == self.data_reader_types.clinical_feature:
            data_reader = FeatureClinicalDataFrameReader
        elif data_type == self.data_reader_types.patho_feature:
            data_reader = FeaturePathologicalDataFrameReader
        elif data_type == self.data_reader_types.blood_feature:
            data_reader = FeatureBloodDataFrameReader
        elif data_type == self.data_reader_types.icd_codes_feature:
            data_reader = FeatureICDCodesDataFrameReader
        elif data_type == self.data_reader_types.tma_cell_density_feature:
            data_reader = FeatureTMACellDensityDataFrameReader

        return data_reader

    def _make_targets_data_frame_reader(
        self, data_type: str, data_reader: DataFrameReader
    ) -> DataFrameReader:
        """Checks the data_type and returns the appropriate reference to the
        data reader class. If there is not match between the data_type and 
        the data related data_type's from the DataReaderTypes class, the
        input data_reader is returned.

        Args:
            data_type (str): Data type of the data reader that should be returned.
            data_reader (DataFrameReader): Reference to a data reader class.

        Returns:
            DataFrameReader: Reference to a data reader class based on the 
            data type.
        """
        if data_type == self.data_reader_types.targets:
            data_reader = TargetsDataFrameReader

        return data_reader

    def _make_data_split_data_frame_reader(
        self, data_type: str, data_reader: DataFrameReader
    ) -> DataFrameReader:
        """Checks the data_type and returns the appropriate reference to the
        data reader class. If there is not match between the data_type and 
        the data related data_type's from the DataReaderTypes class, the
        input data_reader is returned.

        Args:
            data_type (str): Data type of the data reader that should be returned.
            data_reader (DataFrameReader): Reference to a data reader class.

        Returns:
            DataFrameReader: Reference to a data reader class based on the 
            data type.
        """
        if data_type == self.data_reader_types.data_split_blood:
            data_reader = DataSplitBloodDataFrameReader
        elif data_type == self.data_reader_types.data_split_clinical:
            data_reader = DataSplitClinicalDataFrameReader
        elif data_type == self.data_reader_types.data_split_patho:
            data_reader = DataSplitPathologicalDataFrameReader
        elif data_type == self.data_reader_types.data_split_in:
            data_reader = DataSplitInDataFrameReader
        elif data_type == self.data_reader_types.data_split_orypharynx:
            data_reader = DataSplitOrypharynxDataFrameReader
        elif data_type == self.data_reader_types.data_split_out:
            data_reader = DataSplitOutDataFrameReader
        elif data_type == self.data_reader_types.data_split_treatment_outcome:
            data_reader = DataSplitTreatmentOutcomeDataFrameReader

        return data_reader
