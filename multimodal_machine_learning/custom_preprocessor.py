from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from feature_extraction.extract_tabular_features import (
    NOMINAL_FEATURES, ORDINAL_FEATURES, DISCRETE_FEATURES, NUMERICAL_INTEGER_FEATURES
)
from feature_extraction.extract_tabular_features import BLOOD_FEATURES
from feature_extraction.extract_tma_features import TMA_FEATURES
from data_reader import DataFrameReaderFactory
# from multimodal_machine_learning import PredictionPlotter


class ColumnPreprocessor(ColumnTransformer):
    def __init__(self, columns: list[str], min_max_scaler: bool = False):
        self.columns = columns
        self.min_max_scaler = min_max_scaler
        categorical_columns = NOMINAL_FEATURES
        numeric_columns = BLOOD_FEATURES + TMA_FEATURES + DISCRETE_FEATURES + ORDINAL_FEATURES

        self.categorical_columns = [col for col in categorical_columns if col in columns]
        self.numeric_columns = [col for col in numeric_columns if col in columns]
        self.numerical_integer_columns = [
            col for col in columns
            if col in NUMERICAL_INTEGER_FEATURES
        ]

        pipeline_categorical = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])
        pipeline_numeric = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler() if min_max_scaler else StandardScaler())
        ])
        pipeline_encoded = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers = [
            ("categorical", pipeline_categorical, self.categorical_columns),
            ("numeric", pipeline_numeric, self.numeric_columns),
            ("encoded", pipeline_encoded, self.numerical_integer_columns)
        ]

        super().__init__(
            transformers=transformers,
            remainder="passthrough",
            verbose=False
        )



