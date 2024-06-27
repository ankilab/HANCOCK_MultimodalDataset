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
    TargetsDataFRameReader
)
import warnings


class DataFrameReaderFactory:
    """The DataReaderFactory class creates the appropriate DataReader object based on the 
    data type provided by the user.
    """

    def __init__(self):
        pass

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
            DataFrameReader: _description_
        """
        if data_dir_flag:
            if data_type == 'Pathological':
                return PathologicalDataFrameReader(data_dir)
            elif data_type == 'Clinical':
                return ClinicalDataFrameReader(data_dir)
            elif data_type == 'Blood':
                return BloodDataFrameReader(data_dir)
            elif data_type == 'WSI_PrimaryTumor':
                return WSIPrimaryTumorDataFrameReader(data_dir)
            elif data_type == 'WSI_LymphNode':
                return WSILymphNodeDataFrameReader(data_dir)
            elif data_type == 'TMA_CellDensityMeasurement':
                return TMACellDensityDataFrameReader(data_dir)
            elif data_type == 'TextData_reports':
                return TextDataReportsDataFrameReader(data_dir)
            elif data_type == 'Clinical Feature':
                return FeatureClinicalDataFrameReader(data_dir)
            elif data_type == 'Pathological Feature':
                return FeaturePathologicalDataFrameReader(data_dir)
            elif data_type == 'Blood Feature':
                return FeatureBloodDataFrameReader(data_dir)
            elif data_type == 'ICD Codes Feature':
                return FeatureICDCodesDataFrameReader(data_dir)
            elif data_type == 'Feature TMA Cell Density':
                return FeatureTMACellDensityDataFrameReader(data_dir)
            elif data_type == 'Targets':
                return TargetsDataFRameReader(data_dir) 
        else:
            if data_type == 'Pathological':
                return PathologicalDataFrameReader()
            elif data_type == 'Clinical':
                return ClinicalDataFrameReader()
            elif data_type == 'Blood':
                return BloodDataFrameReader()
            elif data_type == 'WSI_PrimaryTumor':
                return WSIPrimaryTumorDataFrameReader()
            elif data_type == 'WSI_LymphNode':
                return WSILymphNodeDataFrameReader()
            elif data_type == 'TMA_CellDensityMeasurement':
                return TMACellDensityDataFrameReader()
            elif data_type == 'TextData_reports':
                return TextDataReportsDataFrameReader()
            elif data_type == 'Clinical Feature':
                return FeatureClinicalDataFrameReader()
            elif data_type == 'Pathological Feature':
                return FeaturePathologicalDataFrameReader()
            elif data_type == 'Blood Feature':
                return FeatureBloodDataFrameReader()
            elif data_type == 'ICD Codes Feature':
                return FeatureICDCodesDataFrameReader()
            elif data_type == 'Feature TMA Cell Density':
                return FeatureTMACellDensityDataFrameReader()
            elif data_type == 'Targets':
                return TargetsDataFRameReader(data_dir) 
        warnings.warn('The data type is not recognized and thus the default'
                    + ' data reader used.')
        return DataFrameReader(self.data_dir)
