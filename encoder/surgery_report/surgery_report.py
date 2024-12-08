# ====================================================================================================================
# Imports
# ====================================================================================================================
from transformers import AutoTokenizer, AutoModel
from openai import AzureOpenAI
import json
import nltk
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path, PosixPath
import sys
sys.path.append(str(Path(__file__).parents[2]))
from encoder.base_encoder import EncoderModel
from encoder.base_encoder import Encoder
from data_reader import DataFrameReaderFactory
from defaults import (
    DefaultPaths, DefaultNames
)
from encoder.storage_handler import NumpyTensorHandler, ChromaTensorHandler
from data_exploration.plot_available_data import HancockAvailableDataPlotter
from encoder.similarity_measure import (
    EncoderSimilarityComparer,
    CosineSimilarity
)


# ====================================================================================================================
# BERT Encoder Implementations
# ====================================================================================================================
class BertEncoder:
    """
    An implementation for encoding textual information using a BERT model. Class is ready to use as defined here.
    """
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._max_length = 512

    def encode(self, text_batch: list[str]) -> np.ndarray:
        """
        Encode a batch of sentences into a single embedding vector. Every sentence is individually tokenized and then
        chunked to fit into the model's maximum input length. The embeddings of the CLS tokens are averaged over the
        whole text_batch to return a single embedding vector.

        Args:
            text_batch (list[str]): A list of sentences that should be encoded into a single embedding vector.
        """
        cls_embedding, text_batch_chunked = self.encode_all(text_batch)
        return cls_embedding[0]

    def encode_all(self, text_batch: list[str]) -> np.ndarray | None:
        nltk.download('punkt')
        nltk.download('punkt_tab')

        text_batch_chunked = self._prepare_text_batch(text_batch=text_batch)
        tokens = self.tokenizer(text_batch_chunked, return_tensors='pt', padding=True)
        with torch.no_grad():
            output = self.model(**tokens)
        embeddings = output.last_hidden_state
        cls_embedding = np.array(embeddings[:, 0, :])
        cls_embedding_combined = self._combine_information(cls_embedding)
        cls_embedding = np.vstack((cls_embedding_combined, cls_embedding))
        print(f'Cls embedding shape: {cls_embedding.shape}')
        text_batch_chunked.insert(0, '')
        return [cls_embedding, text_batch_chunked]

    @staticmethod
    def _combine_information(cls_embedding) -> np.ndarray:
        return np.average(cls_embedding, axis=0)

    def _prepare_text_batch(self, text_batch: list[str]) -> list[str]:
        output_text_batch = []
        for text in text_batch:
            token = self.tokenizer(text)
            if len(token['input_ids']) <= self._max_length:
                output_text_batch.append(text)
            else:
                text_chunks = self._split_text_for_max_length(text)
                for text_chunk in text_chunks:
                    output_text_batch.append(text_chunk)
        return output_text_batch

    def _split_text_for_max_length(self, text: str) -> list[str]:
        sentences = nltk.sent_tokenize(text)
        start_index = 0
        output_text = []
        while start_index < len(sentences):
            combined_text, start_index = self._combine_sentences_for_max_length(sentences, start_index)
            output_text.append(combined_text)
        return output_text

    def _combine_sentences_for_max_length(self, sentences: list[str], start_index: int) -> [str, int]:
        for i in reversed(range(start_index, len(sentences) + 1)):
            combined_text = " ".join(sentences[start_index:i])
            token = self.tokenizer(combined_text)
            if len(token['input_ids']) <= self._max_length:
                return [combined_text, i]


class BioClinicalBertEncoder(BertEncoder):
    def __init__(self):
        super().__init__(model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=512)


class PubMedBertEncoder(BertEncoder):
    def __init__(self):
        super().__init__(model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', max_length=512)


# ====================================================================================================================
# OpenAI Encoder Implementations using Microsoft AI Portal
# ====================================================================================================================
def load_environment_variable(env_file: Path = Path(__file__).parents[1] / 'env.json'):
    if env_file.exists():
        try:
            with open(env_file, 'r') as file:
                data = json.load(file)
                for item in data:
                    name = item['name']
                    value = item['value']
                    os.environ[name] = value
        except Exception as ex:
            print('Could not load env variables')
    else:
        os.environ['AZURE_AI_PORTAL_KEY'] = 'No key found'
        os.environ['AZURE_AI_PORTAL_ENDPOINT'] = 'No endpoint found'
        os.environ['AZURE_AI_EMBEDDING_MODEL_ID'] = 'No model ID found'
load_environment_variable()


class AzureAIPortalInterface:
    """Class to interact with the Azure AI Portal. 
    """
    def __init__(
        self, key: str = os.getenv('AZURE_AI_PORTAL_KEY'),
        endpoint: str = os.getenv("AZURE_AI_PORTAL_ENDPOINT"),
        embedding_model_id: str = os.getenv("AZURE_AI_EMBEDDING_MODEL_ID"),
        api_version: str = "2024-02-01"
    ):
        """Class to interact with the Azure AI Portal.

        Args:
            key (str, optional): Key for accessing the Azure AI Portal. 
            Defaults to os.getenv('AZURE_AI_PORTAL_KEY').
            
            endpoint (str, optional): Endpoint for accessing the Azure AI Portal. 
            Defaults to os.getenv("AZURE_AI_PORTAL_ENDPOINT").
            
            embedding_model_id (str, optional): The given name of a deployed 
            embedding model in the Azure AI Portal. 
            Defaults to os.getenv("AZURE_AI_EMBEDDING_MODEL_ID").
            
            api_version (str, optional): The API version to use to connect 
            to the Azure AI Portal. Defaults to "2024-02-01".
        """
        try:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint, api_key=key, api_version=api_version
            )
        except Exception as ex:
            print('Could not connect to Azure AI Portal')
            self.client = None
        self._embedding_model_id = embedding_model_id

    def create_embeddings(self, text: str) -> np.ndarray | None:
        """Creates an embedding for the given text.

        Args:
            text (str): The text that should be embedded.

        Returns:
            list: The embedding as a list of floats.
        """
        if self.client:
            response = self.client.embeddings.create(
                input=text, model=self._embedding_model_id
            )
            return np.array(response.data[0].embedding)
        else:
            return None
    
    @staticmethod
    def _check_parameter_validity(key, endpoint, embedding_model_id) -> None:
        if key is None:
            raise ValueError('Set the key variable needed for the text-embedding-3-large model either manually, ' +
                             'or make sure that the environment is configured appropriately.')
        if endpoint is None:
            raise ValueError('Set the endpoint variable needed for the text-embedding-3-large model either manually, ' +
                             'or make sure that the environment is configured appropriately.')
        if embedding_model_id is None:
            raise ValueError('Set the model ID variable needed for the text-embedding-3-large model either manually, ' +
                             'or make sure that the environment is configured appropriately.')

    def encode(self, text_batch: list[str]) -> np.ndarray | None:
        raise NotImplementedError('This method should not be used as it creates costs.')
        # full_text = ''
        # for text in text_batch:
        #     full_text += text
        # return self.create_embeddings(full_text)


# ====================================================================================================================
# Surgery Report Encoder Model implementation
# ====================================================================================================================
class SurgeryReportEncoderModel(EncoderModel):
    """Encoder model for surgery reports"""
    @property
    def name(self) -> str:
        return 'Surgery Report Encoder'

    def __init__(self, df_data_to_encode: pd.DataFrame = None):
        super().__init__(df_data_to_encode)
        self._surgery_report_encoder_implementation = self._create_encoder_implementation()

    def _create_encoder_implementation(self) -> any:
        """
        Create the encoder implementation that should be used for the Surgery Reports.

        Returns:
            any: The encoder implementation object.
        """
        return BioClinicalBertEncoder()

    def _encode_row(self, row: pd.Series) -> np.ndarray | None:
        surgery_history_path = row['Surgery history'].values[0]
        surgery_description_path = row['Surgery description'].values[0]
        surgery_report_path = row['Surgery report'].values[0]

        surgery_history = None
        if type(surgery_history_path) == PosixPath and surgery_history_path.exists():
                with open(surgery_history_path, 'r') as file:
                    surgery_history = file.read()

        surgery_description = None
        if type(surgery_description_path) == PosixPath and surgery_description_path.exists():
            with open(surgery_description_path, 'r') as file:
                surgery_description = file.read()

        surgery_report = None
        if type(surgery_report_path) == PosixPath and surgery_report_path.exists():
            with open(surgery_report_path, 'r') as file:
                surgery_report = file.read()

        surgery_text = []
        if surgery_history:
            surgery_text.append('Surgery History: ' + surgery_history)
        if surgery_description:
            surgery_text.append('Surgery Description: ' + surgery_description)
        if surgery_report:
            surgery_text.append('Surgery Report: ' + surgery_report)

        if len(surgery_text) == 0:
            return None

        row_encoding = self._surgery_report_encoder_implementation.encode(surgery_text)

        return row_encoding


class SurgeryReportEncoderModelBioClinicalBert(SurgeryReportEncoderModel):
    @property
    def name(self) -> str:
        return 'Bio Clinical BERT'

    def _create_encoder_implementation(self) -> BioClinicalBertEncoder:
        """
        Create the encoder implementation that should be used for the Surgery Reports.

        Returns:
            BioClinicalBertEncoder: The encoder implementation object.
        """
        return BioClinicalBertEncoder()


class SurgeryReportEncoderModelPubMedBert(SurgeryReportEncoderModel):
    @property
    def name(self) -> str:
        return 'PubMed BERT'

    def _create_encoder_implementation(self) -> PubMedBertEncoder:
        """
        Create the encoder implementation that should be used for the Surgery Reports.

        Returns:
            PubMedBertEncoder: The encoder implementation object.
        """
        return PubMedBertEncoder()


class SurgeryReportEncoderModelTextEmbedding3Large(SurgeryReportEncoderModel):
    @property
    def name(self) -> str:
        return 'Text Embedding 3 Large'

    def _create_encoder_implementation(self) -> AzureAIPortalInterface:
        """
        Create the encoder implementation that should be used for the Surgery Reports.

        Returns:
            AzureAIPortalInterface: The encoder implementation object.
        """
        load_environment_variable()
        return AzureAIPortalInterface()


# ====================================================================================================================
# Surgery Report Encoder
# ====================================================================================================================
class SurgeryReportEncoder(Encoder):
    def __init__(self):
        data_frame_reader_factory = DataFrameReaderFactory()
        default_paths = DefaultPaths()
        default_names = DefaultNames()
        file_extension = '.npy'
        encoding_base_name = default_names.encodings_surgery_report
        raw_data_location_reader = data_frame_reader_factory.make_data_frame_reader(
            data_frame_reader_factory.data_reader_types.text_reports_merged_english,
            data_dir=default_paths.text_data_dir, data_dir_flag=True
        )
        encoding_base_save_dir = default_paths.encodings_base
        super().__init__(
            raw_data_location_reader=raw_data_location_reader,
            encoding_base_save_dir=encoding_base_save_dir,
            encoding_model_class=SurgeryReportEncoderModel,
            encoding_file_name=encoding_base_name,
            encoding_file_extension=file_extension,
            encoding_storage_handler=NumpyTensorHandler()
        )


class SurgeryReportEncoderBioClinicalBert(Encoder):
    def __init__(self):
        data_frame_reader_factory = DataFrameReaderFactory()
        default_paths = DefaultPaths()
        default_names = DefaultNames()
        file_extension = '.npy'
        encoding_storage_handler = NumpyTensorHandler()
        encoding_base_save_dir = default_paths.encodings_base
        # file_extension = '.db'
        # encoding_storage_handler = ChromaTensorHandler()
        # encoding_base_save_dir = default_paths.encodings_db_path

        encoding_base_name = default_names.encodings_surgery_report_bio_clinical_bert
        raw_data_location_reader = data_frame_reader_factory.make_data_frame_reader(
            data_frame_reader_factory.data_reader_types.text_reports_merged_english,
            data_dir=default_paths.text_data_dir, data_dir_flag=True
        )
        super().__init__(
            raw_data_location_reader=raw_data_location_reader,
            encoding_base_save_dir=encoding_base_save_dir,
            encoding_model_class=SurgeryReportEncoderModelBioClinicalBert,
            encoding_file_name=encoding_base_name,
            encoding_file_extension=file_extension,
            encoding_storage_handler=encoding_storage_handler
        )


class SurgeryReportEncoderPubMedBert(Encoder):
    def __init__(self):
        data_frame_reader_factory = DataFrameReaderFactory()
        default_paths = DefaultPaths()
        default_names = DefaultNames()

        file_extension = '.npy'
        encoding_storage_handler = NumpyTensorHandler()
        encoding_base_save_dir = default_paths.encodings_base
        # file_extension = '.db'
        # encoding_storage_handler = ChromaTensorHandler()
        # encoding_base_save_dir = default_paths.encodings_db_path

        encoding_base_name = default_names.encodings_surgery_report_pub_med_bert
        raw_data_location_reader = data_frame_reader_factory.make_data_frame_reader(
            data_frame_reader_factory.data_reader_types.text_reports_merged_english,
            data_dir=default_paths.text_data_dir, data_dir_flag=True
        )
        super().__init__(
            raw_data_location_reader=raw_data_location_reader,
            encoding_base_save_dir=encoding_base_save_dir,
            encoding_model_class=SurgeryReportEncoderModelPubMedBert,
            encoding_file_name=encoding_base_name,
            encoding_file_extension=file_extension,
            encoding_storage_handler=encoding_storage_handler
        )


class SurgeryReportEncoderTextEmbedding3Large(Encoder):
    def __init__(self):
        data_frame_reader_factory = DataFrameReaderFactory()
        default_paths = DefaultPaths()
        default_names = DefaultNames()

        file_extension = '.npy'
        encoding_storage_handler = NumpyTensorHandler()
        encoding_base_save_dir = default_paths.encodings_base
        # file_extension = '.db'
        # encoding_storage_handler = ChromaTensorHandler()
        # encoding_base_save_dir = default_paths.encodings_db_path

        encoding_base_name = default_names.encodings_surgery_report_text_embedding_3_large
        raw_data_location_reader = data_frame_reader_factory.make_data_frame_reader(
            data_frame_reader_factory.data_reader_types.text_reports_merged_english,
            data_dir=default_paths.text_data_dir, data_dir_flag=True
        )
        super().__init__(
            raw_data_location_reader=raw_data_location_reader,
            encoding_base_save_dir=encoding_base_save_dir,
            encoding_model_class=SurgeryReportEncoderModelTextEmbedding3Large,
            encoding_file_name=encoding_base_name,
            encoding_file_extension=file_extension,
            encoding_storage_handler=encoding_storage_handler
        )


# ====================================================================================================================
# Surgery Data Plotter
# ====================================================================================================================
class SurgeryDataPlotter(HancockAvailableDataPlotter):
    """
    Plots the available data from the hancock dataset for the
    surgery reports.
    """
    def _get_available_data(self) -> pd.DataFrame:
        """
        Overwrites the method from the parent class to use the
        surgery report data instead of the whole data set.
        """
        data_factory = DataFrameReaderFactory()
        data_reader = data_factory.make_data_frame_reader(
            data_type=data_factory.data_reader_types.text_reports_merged_english
        )
        data = data_reader.return_data_count()
        data = data.fillna(0)
        columns_to_convert = ['Surgery history', 'Surgery description', 'Surgery report']
        data[columns_to_convert] = data[columns_to_convert].astype(int)
        return data


# ====================================================================================================================
# Surgery Encoding Execution
# ====================================================================================================================
def encode_surgery_text_data() -> dict[str, np.ndarray]:
    encoder_bio = SurgeryReportEncoderBioClinicalBert()
    encoder_pub = SurgeryReportEncoderPubMedBert()
    print('Encode using Bio Bert')
    bio_encodings = encoder_bio.return_and_save_all_encodings()
    print('Encode using Pub Bert')
    pub_encodings = encoder_pub.return_and_save_all_encodings(force_new_encoding_creation=False)

    encoder_3_large = SurgeryReportEncoderTextEmbedding3Large()
    print('E(ncode using Text Embedding 3 Large')
    large_encodings = encoder_3_large.return_and_save_all_encodings()
    return {
        'Bio encodings': bio_encodings,
        'Pub encodings': pub_encodings,
        'Large encodings': large_encodings
    }


# ====================================================================================================================
# Fun
# ====================================================================================================================
if __name__ == '__main__':
    print('Executing surgery report.py')
    encoder_list = [
        SurgeryReportEncoderBioClinicalBert(),
        SurgeryReportEncoderPubMedBert(),
        SurgeryReportEncoderTextEmbedding3Large()
    ]

    sim_comparer = EncoderSimilarityComparer(
        encoder_list=encoder_list,
        similarity_measure_cls=CosineSimilarity,
        image_base_name="surgery_text"
    )
    sim_comparer.visualize_similarity_inter_encoder_based_all_relevant_features()


