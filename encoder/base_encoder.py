# ====================================================================================================================
# Imports
# ====================================================================================================================
from typing import Type
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_reader import DataFrameReader, DataFrameReaderFactory
from encoder.storage_handler import (
    NumpyTensorHandler,
    DataHandler
)
from defaults import DefaultPaths, DefaultNames


# ====================================================================================================================
# Abstract Encoder Model
# ====================================================================================================================
class EncoderModel:
    """Abstract class for an encoder model. Should be implemented individually for each encoder model.

    Public Methods:
        encode: Creates an encoding for a given patient ID if it exists on the DataFrame and is unique.
        encode_all: Creates encodings for all valid patient IDs in the DataFrame.

    Abstract Methods:
        _encode_row: Must be implemented by the subclass. Creates an encoding for a given row.

    Abstract Properties:
        name: Must be implemented by the subclass. The name of the encoder model.

    Properties:
        df_data: Get a copy of the DataFrame containing the file paths.
    """
    @property
    def name(self) -> str:
        raise NotImplementedError('This property must be implemented by the subclass')

    @property
    def df_data(self) -> pd.DataFrame:
        """
        Get a copy of the DataFrame containing the file paths.
        """
        return self._df_data.copy()

    def __init__(self, df_data_to_encode: pd.DataFrame):
        """
        Initializes the Encoder model.

        Args:
            df_data_to_encode (pd.DataFrame): A DataFrame containing the data that should be encoded. Can
            be either the file path or the actual data.
        """
        self._df_data = df_data_to_encode

    # ================================================================================================================
    # Public Methods
    # ================================================================================================================
    def encode(self, patient_id: str) -> np.ndarray | None:
        """
        Creates an encoding for a given patient ID if it exists on the DataFrame and is unique.

        Args:
            patient_id (str): The patient ID to encode. Should be in form of '001', '002', etc.

        Returns:
            np.ndarray | None: If the patient ID is valid, returns the encoding. Otherwise, returns None.
        """
        relevant_row = self._df_data[self._df_data['patient_id'] == patient_id]
        if not self._row_is_valid(relevant_row, patient_id):
            return None

        try:
            encoding = self._encode_row(relevant_row)
        except Exception as ex:
            print(f'Exception occurred while encoding patient ID {patient_id}: {ex}')
            return None

        return encoding

    def encode_all(self) -> dict[str: np.ndarray]:
        """
        Creates encodings for all valid patient IDs in the DataFrame.

        Returns:
            dict[str: np.ndarray]: A dictionary containing the patient IDs as keys and the encodings as values.
        """
        patient_ids = self._df_data['patient_id']
        encodings_dict = {}
        for patient_id in patient_ids:
            encoding = self.encode(patient_id=patient_id)
            if encoding is not None:
                encodings_dict[patient_id] = encoding
        return encodings_dict

    # ================================================================================================================
    # Abstract Methods
    # ================================================================================================================
    def _encode_row(self, row: pd.Series) -> any:
        raise NotImplementedError('This method must be implemented by the subclass')

    # ================================================================================================================
    # Static Methods
    # ================================================================================================================
    @staticmethod
    def _row_is_valid(row: pd.Series, patient_id: str) -> bool:
        """
        Check if the row is a valid for and is not empty or duplicated.

        Args:
            row (pd.Series): The row to check.
            patient_id (str): The patient ID for the row.

        Returns:
            bool: True if the row is valid, False otherwise.
        """
        if len(row) == 0:
            warnings.warn(f'Patient ID {patient_id} not found in the DataFrame. '
                          f'No encoding will be returned. '
                          f'Please ensure that the patient ID is in the DataFrame.')
            return False
        elif len(row) != 1:
            warnings.warn(f'Patient ID {patient_id} is not unique in the DataFrame.'
                          f'Please ensure that the DataFrame has unique patient IDs. '
                          f'This patient will be ignored and no encoding will be returned.')
            return False
        return True


# ====================================================================================================================
# Base Encoder
# ====================================================================================================================
class Encoder:
    """
    Encoder object that should be used to create, receive and return encodings. The Encoder uses a EncoderModel
    object to create the encodings

    Public Methods:
        return_and_save_encoding: Returns the encoding for a given patient ID. If the encoding does not exist already
            at the expected location, it will try to create a new encoding for the patient ID.
        read_encoding: Tries to read the encoding from the expected location.
        create_encoding: Tries to create the encoding for the given patient ID.
        save_encoding: Tries to save the encoding to the expected location for the given patient ID.
        return_and_save_all_encodings: Tries to create or read the encodings for all patients in the DataFrame.

    Attributes:
        raw_data_location_reader (DataFrameReader): The DataFrameReader object that will be used to read the raw data
            and contain the raw data paths or the raw data itself.
        encoding_base_save_dir (Path): The base directory where the encodings will be saved.
        encoding_base_name (str): The base file name for the encoding files.
        encoding_storage_handler (DataHandler): The DataHandler object that will be used to save and load the encodings.
        encoding_file_extension (str): The file extension for the encoding files. Should be compatible with the
            encoding_storage_handler.
        encoder_model (EncoderModel): The EncoderModel object that will be used to create the encodings.
    """
    def __init__(
            self,
            raw_data_location_reader: DataFrameReader,
            encoding_base_save_dir: Path,
            encoding_model_class: Type[EncoderModel] = EncoderModel,
            encoding_file_name: str = 'test_encoding',
            encoding_file_extension: str = '.npy',
            encoding_storage_handler: DataHandler = NumpyTensorHandler
    ):
        """
        Creates a new Encoder object that should be used to create, receive and save encodings. The Encoder object
        is intended to work on modality specific data, such as surgery reports, TMA Slides or Tissue Slides.

        Args:
            raw_data_location_reader (DataFrameReader): The DataFrameReader object that will be used to read the raw data
            and contain the raw data paths.

            encoding_base_save_dir (Path): The base directory where the encodings will be saved.

            encoding_model_class (Type[EncoderModel]): The class that will be used to create the encoding model. This should
            be a subclass of EncoderModel.

            encoding_file_name (str): The file name for the encoding files.

            encoding_file_extension (str): The file extension for the encoding files. Should be compatible with the
            encoding_storage_handler.

            encoding_storage_handler (DataHandler): The DataHandler object that will be used to save and load the
            encodings.
        """
        self.raw_data_location_reader = raw_data_location_reader
        self.encoding_base_save_dir = encoding_base_save_dir
        self.encoding_base_name = encoding_file_name
        self.encoding_storage_handler = encoding_storage_handler
        self.encoding_file_extension = encoding_file_extension
        self._check_df_file_paths()
        self.encoder_model = encoding_model_class(self.raw_data_location_reader.return_data())

    # ================================================================================================================
    # Public Methods
    # ================================================================================================================
    def return_and_save_encoding(self, patient_id: str, force_new_encoding_creation: bool = False) -> any:
        """
        Returns the encoding for a given patient ID. If the encoding does not exist already at the expected location,
        it will try to create a new encoding for the patient ID.

        Args:
            patient_id (str): The patient ID for which to return the encoding.

            force_new_encoding_creation (bool, optional): Flag for enforcing the creation of a new encoding. If set to
            True it will not look for an existing encoding and will creat a new one and will overwrite any existing
            encoding. Defaults to False.

        Returns:
            any: The encoding for the patient ID if possible, otherwise None.
        """
        if force_new_encoding_creation:
            encoding = self.create_encoding(patient_id)
        else:
            encoding = self.read_encoding(patient_id)
            if encoding is None:
                encoding = self.create_encoding(patient_id)
            else:
                return encoding

        if encoding is not None:
            print(f"Created and saved encoding for patient ID {patient_id}")
            self.save_encoding(patient_id, encoding)

        return encoding

    def read_encoding(self, patient_id: str) -> any:
        """
        Tries to read the encoding from the expected location.

        Args:
            patient_id (str): The patient ID for which to read the encoding.

        Returns:
            any: The encoding if it exists, otherwise None.
        """
        encoding = self.encoding_storage_handler.load_file(
            base_path=self.encoding_base_save_dir, file_id=patient_id,
            file_name=self.encoding_base_name, file_extension=self.encoding_file_extension
        )
        return encoding

    def create_encoding(self, patient_id: str) -> any:
        """
        Tries to create the encoding for the given patient ID.

        Args:
            patient_id (str): The patient ID for which to create the encoding.

        Returns:
            any: The encoding if it was possible to create, otherwise None.
        """
        encoding = self.encoder_model.encode(patient_id)
        return encoding

    def save_encoding(self, patient_id: str, encoding: any) -> bool:
        """
        Tries to save the encoding to the expected location for the given patient ID.

        Args:
            patient_id (str): The patient ID for which to save the encoding.

            encoding (any): The encoding to save. Should be compatible with the encoding_storage_handler.
        """
        return self.encoding_storage_handler.save_file(
            base_path=self.encoding_base_save_dir, file_id=patient_id,
            file_name=self.encoding_base_name,
            file_extension=self.encoding_file_extension, file=encoding, create_path=True)

    def return_and_save_all_encodings(self, force_new_encoding_creation: bool = False) -> dict[str: any]:
        """
        Tries to create or read the encodings for all patients in the DataFrame.

        Args:
            force_new_encoding_creation (bool, optional): Flag for enforcing the creation of new encodings. If set to
            True it will not look for existing encodings and will create new ones and will overwrite any existing.
            Defaults to False

        Returns:
            dict[str: any]: A dictionary containing the patient IDs as keys and the encodings as values. Does not
            contain patient ID's if it was not possible to create or read an encoding.
        """
        patient_ids = self.raw_data_location_reader.return_data()['patient_id']
        encodings = {}
        for patient_id in patient_ids:
            encoding = self.return_and_save_encoding(patient_id, force_new_encoding_creation)
            if encoding is not None:
                encodings[patient_id] = encoding

        return encodings

    # ================================================================================================================
    # Private Helper Methods
    # ================================================================================================================
    def _check_df_file_paths(self):
        data = self.raw_data_location_reader.return_data()
        try:
            patient_ids = data['patient_id']
        except Exception as ex:
            raise KeyError('DataFrame must have a column named "patient_id"')

    def _create_encoding_path(self, patient_id: str) -> Path:
        encoding_path = (self.encoding_base_save_dir / patient_id /
                         f'{self.encoding_base_name}{self.encoding_file_extension}')
        return encoding_path


# ====================================================================================================================
# Helper Classes
# ====================================================================================================================
class EncodingDataFrameCreator:
    """
    Helper class that creates a single dataframe for the
    encodings from the given encoder, as well as the clinical and pathological data.
    The encodings are saved in the column 'encoding'.

    Methods:
        create_combined_data_frame: Creates a combined Data Frame with the encodings, clinical and pathological data.

    Attributes:
        encoder (Encoder): The encoder object that is used to create the encodings.
    """
    def __init__(
            self,
            encoder: Encoder,
            patho_data_path: Path = DefaultPaths().patho,
            clinical_data_path: Path = DefaultPaths().clinical, **kwargs
    ):
        self._patho_data_path = patho_data_path
        self._clinical_data_path = clinical_data_path
        self.encoder = encoder
        self._data_factory = DataFrameReaderFactory()
        self._check_inserted_paths()

    def _check_inserted_paths(self) -> None:
        if not self._clinical_data_path.exists():
            raise FileNotFoundError(f'Clinical data file {self._clinical_data_path} does not exist')
        if not self._patho_data_path.exists():
            raise FileNotFoundError(f'Pathological data file {self._patho_data_path} does not exist')

    def create_combined_data_frame(self) -> pd.DataFrame:
        """
        Creates a combined Data Frame with the encodings, clinical and pathological data.

        Returns:
            pd.DataFrame: The combined Data Frame.
        """
        clinical_data = self._data_factory.make_data_frame_reader(
            data_type=self._data_factory.data_reader_types.clinical, data_dir=self._clinical_data_path,
            data_dir_flag=True
        ).return_data()
        patho_data = self._data_factory.make_data_frame_reader(
            data_type=self._data_factory.data_reader_types.patho, data_dir=self._patho_data_path,
            data_dir_flag=True
        ).return_data()
        encodings = self.encoder.return_and_save_all_encodings()
        encodings = pd.DataFrame(
            {
                'patient_id': encodings.keys(),
                DefaultNames().encoding_column_name: encodings.values()
            }
        )
        merged_data = encodings.merge(clinical_data, on='patient_id', how='inner')
        merged_data = merged_data.merge(patho_data, on='patient_id', how='inner')
        return merged_data
