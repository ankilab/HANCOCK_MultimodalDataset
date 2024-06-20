from pathlib import Path
from data_reader.data_reader import (
    DataFrameReader,
    PathologicalDataFrameReader,
    ClinicalDataFrameReader,
    BloodDataFrameReader
)

class DataFrameReaderFactory:
    """The DataReaderFactory class creates the appropriate DataReader object based on the 
    data type provided by the user.
    """
    def __init__(self):
        pass
        
    def make_data_frame_reader(self, data_type:str = 'NA', data_dir:Path = Path(__file__)) -> DataFrameReader:
        if data_type == 'Pathological':
            return PathologicalDataFrameReader(data_dir)
        elif data_type == 'Clinical':
            return ClinicalDataFrameReader(data_dir)
        elif data_type == 'Blood':
            return BloodDataFrameReader(data_dir)
        elif data_type == 'WSI_PrimaryTumor':
            return DataFrameReader(data_dir)
        elif data_type == 'WSI_LymphNode':
            return DataFrameReader(data_dir)
        elif data_type == 'TMA_CellDensityMeasurement':
            return DataFrameReader(data_dir)
        elif data_type == 'TextData_reports':
            return DataFrameReader(data_dir)
        else:
            return DataFrameReader(self.data_dir)