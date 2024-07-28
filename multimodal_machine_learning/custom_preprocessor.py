from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from feature_extraction.extract_tabular_features import (
    NOMINAL_FEATURES, ORDINAL_FEATURES, DISCRETE_FEATURES, NUMERICAL_INTEGER_FEATURES
)
from feature_extraction.extract_tabular_features import BLOOD_FEATURES
from feature_extraction.extract_tma_features import TMA_FEATURES


TMA_VECTOR_FEATURES = [
    'HE', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'MHC1', 'PDL1'
]
TMA_VECTOR_LENGTH = 512


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
        self.tma_vector_columns = [col for col in columns
                                   if col in TMA_VECTOR_FEATURES]

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

    def fit(self, x: pd.DataFrame, y: any = None, **params) -> ColumnTransformer:
        super().fit(x, y, **params)
        return self

    def transform(self, x: pd.DataFrame, **params) -> np.ndarray:
        x_transformed = super().transform(x, **params)
        x_transformed = pd.DataFrame(x_transformed, columns=self.get_feature_names_out())

        for col in self.tma_vector_columns:
            col_name = 'remainder__' + col
            x_transformed[col_name] = x_transformed[col_name].apply(
                lambda row:
                np.zeros(512)
                if isinstance(row, float) and np.isnan(row)
                else row
            )

            split_columns = pd.DataFrame(
                np.vstack(x_transformed[col_name]),
                index=x_transformed.index,
                columns=[f'{i}_{col}' for i in range(TMA_VECTOR_LENGTH)])

            x_transformed = pd.concat([x_transformed, split_columns], axis=1)
            x_transformed.drop(col_name, axis=1, inplace=True)

        return x_transformed.values
