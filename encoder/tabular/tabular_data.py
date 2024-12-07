# ====================================================================================================================
# Imports
# ====================================================================================================================
import sys
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.append(str(Path(__file__).parents[2]))
from encoder.storage_handler import NumpyTensorHandler
from data_reader import DataFrameReaderFactory
from defaults import DefaultPaths, DefaultNames
from encoder.base_encoder import EncoderModel, Encoder
from multimodal_machine_learning.custom_preprocessor import HancockTabularPreprocessor


# ====================================================================================================================
# Dummy class for Tabular Merged Encoder Model
# ====================================================================================================================
class TabularMergedEncoderModel(EncoderModel):
    @property
    def name(self) -> str:
        return 'Tabular Merged Encoder'

    def __init__(self, df_data_to_encode: pd.DataFrame):
        super().__init__(df_data_to_encode)
        self.preprocessor = HancockTabularPreprocessor(self.df_data.columns[1:])
        self.encodings = self.preprocessor.fit_transform(self.df_data.drop('patient_id', axis=1))

    def _encode_row(self, row: pd.Series) -> any:
        encoding = self.preprocessor.transform(row.drop('patient_id', axis=1))[0]
        return encoding


# ====================================================================================================================
# Dummy class for Tabular Merged Encoder
# ====================================================================================================================
class TabularMergedEncoder(Encoder):
    def __init__(self, ):
        factory = DataFrameReaderFactory()
        default_paths = DefaultPaths()
        default_names = DefaultNames()
        data_reader = factory.make_data_frame_reader(
            data_type=factory.data_reader_types.tabular_merged_feature
        )
        encoding_base_save_dir = default_paths.encodings_base
        encoding_file_name = default_names.encodings_tabular_merged
        super().__init__(
            raw_data_location_reader=data_reader,
            encoding_base_save_dir=encoding_base_save_dir,
            encoding_model_class=TabularMergedEncoderModel,
            encoding_file_name=encoding_file_name,
            encoding_file_extension='.npy',
            encoding_storage_handler=NumpyTensorHandler()
        )

if __name__ == '__main__':
    encoder = TabularMergedEncoder()
    encoder.return_and_save_all_encodings(force_new_encoding_creation=True)


