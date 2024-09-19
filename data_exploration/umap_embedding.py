from feature_extraction.extract_tabular_features import NOMINAL_FEATURES, ORDINAL_FEATURES, DISCRETE_FEATURES
from feature_extraction.extract_tabular_features import BLOOD_FEATURES
from feature_extraction.extract_tma_features import TMA_FEATURES

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from umap import UMAP
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


SEED = 42


def setup_preprocessing_pipeline(
        columns: list[str], min_max_scaler: bool = False
) -> ColumnTransformer:
    """
    Sets up a sklearn pipeline for preprocessing the data before applying UMAP
    Parameters
    ----------
    columns : list of str
        List of columns to consider
    min_max_scaler: bool
        If True, MinMaxScaler is used instead of StandardScaler. Default: False

    Returns
    -------
    Scikit-learn ColumnTransformer

    """
    categorical_columns = NOMINAL_FEATURES
    numeric_columns = BLOOD_FEATURES + TMA_FEATURES + DISCRETE_FEATURES + ORDINAL_FEATURES

    categorical_columns = [col for col in categorical_columns if col in columns]
    numeric_columns = [col for col in numeric_columns if col in columns]

    remaining_columns = [col for col in columns if col not in categorical_columns and col not in numeric_columns]
    pipeline_categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    pipeline_numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler() if min_max_scaler else StandardScaler())
    ])
    pipeline_encoded = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", pipeline_categorical, categorical_columns),
            ("numeric", pipeline_numeric, numeric_columns),
            ("encoded", pipeline_encoded, remaining_columns)
        ],
        remainder="passthrough",
        verbose=False
    )
    return preprocessor


def get_umap_embedding(features_directory, umap_min_dist=0.1, umap_n_neighbors=15):
    """
    Loads multimodal features, preprocessed them and applies UMAP to reduce the dimensions.
    Parameters
    ----------
    features_directory : str
        Directory that contains the extracted features (.csv)
    umap_min_dist : flaot
        min_dist parameter for UMAP
    umap_n_neighbors : int
        n_neighors parameter for UMAP

    Returns
    -------
    Pandas DataFrame
        Dataframe containing the features and the resulting embedding in the "UMAP 1" and "UMAP 2" columns

    """
    fdir = Path(features_directory)

    # Load encoded data
    clinical = pd.read_csv(fdir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(fdir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(fdir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(fdir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density= pd.read_csv(fdir/"tma_cell_density.csv", dtype={"patient_id": str})

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="inner")
    df = df.merge(blood, on="patient_id", how="inner")
    df = df.merge(icd, on="patient_id", how="inner")
    df = df.merge(cell_density, on="patient_id", how="inner")
    df = df.reset_index(drop=True)

    # Preprocess embeddings
    preprocessor = setup_preprocessing_pipeline(df.columns[1:])
    embeddings = preprocessor.fit_transform(df.drop("patient_id", axis=1))

    # Reduce to 2D
    umap = UMAP(random_state=SEED, min_dist=umap_min_dist, n_neighbors=umap_n_neighbors).fit_transform(embeddings)

    # Normalize axes
    tx, ty = umap[:, 0], umap[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    # Add UMAP to the dataframe
    df["UMAP 1"] = tx
    df["UMAP 2"] = ty

    return df


def plot_umap(dataframe, subplot_titles, subplot_features, numerical_features=[], marker_size=4, filename=None):
    """
    Shows 2D UMAP embeddings in a scatterplot where points are colored by distinct features.
    Parameters
    ----------
    dataframe : Pandas dataframe
        Dataframe that contains both the UMAP embeddings and features
    subplot_titles : list of str
        List of titles for suplots
    subplot_features : list of str
        List of features (columns in the dataframe)
    numerical_features : list of str
        List of features that are numerical. Default: []
    marker_size : int
        Marker size for scatter plot. Default: 4
    filename : str
        Plot is saved to a file if filename is specified. Default: None

    """
    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, len(subplot_titles), figsize=(7, 2.5))

    for i, feature in enumerate(subplot_features):

        if feature in numerical_features:
            df = dataframe.copy()
            palette = sns.color_palette("plasma", as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(df[feature].min().min(),
                                                                        df[feature].max().max()))
            hue_norm = sm.norm
            legend = False

        else:
            df = dataframe.copy().fillna("missing")
            palette = sns.color_palette("Set2", n_colors=len(df[feature].fillna("missing").unique()))
            hue_norm = None
            legend = True

        sns.scatterplot(df, x="UMAP 1", y="UMAP 2", hue=feature,
                        hue_norm=hue_norm, palette=palette, legend=legend, s=marker_size, ax=axes[i]);

        # Axes
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_aspect("equal")

        # Title
        axes[i].set_title(subplot_titles[i])

        plt.tight_layout(pad=2)
        if feature in numerical_features:
            plt.colorbar(sm, ax=axes[i], fraction=0.06)
        else:
            axes[i].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False, fontsize=6)

    sns.despine()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.show()