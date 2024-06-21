from pathlib import Path
from .data_reader import (
    DataFrameReader,
    PathologicalDataFrameReader,
    ClinicalDataFrameReader,
    BloodDataFrameReader,
    WSIPrimaryTumorDataFrameReader,
    WSILymphNodeDataFrameReader,
    TMACellDensityDataFrameReader,
    TextDataReportsDataFrameReader
)
import warnings


class DataFrameReaderFactory:
    """The DataReaderFactory class creates the appropriate DataReader object based on the 
    data type provided by the user.
    """

    def __init__(self):
        pass

    def make_data_frame_reader(
        self, data_type: str = 'NA', data_dir: Path = Path(__file__)
    ) -> DataFrameReader:
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
        else:
            warnings.warn('The data type is not recognized and thus the default' 
                          +' data reader used.')
            return DataFrameReader(self.data_dir)
