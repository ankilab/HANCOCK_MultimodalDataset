import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


BINARY_FEATURES = [
    "primarily_metastasis",
    "sex",
    "lymphovascular_invasion_L",
    "vascular_invasion_V",
    "perineural_invasion_Pn",
    "perinodal_invasion",
    "carcinoma_in_situ",
]
NOMINAL_FEATURES = [
    "smoking_status",
    "primary_tumor_site",
    "histologic_type",
    "hpv_association_p16",
    "grading",
    "resection_status_carcinoma_in_situ",
    "resection_status"
]
DISCRETE_FEATURES = [
    "age_at_initial_diagnosis",
    "number_of_positive_lymph_nodes",
    "infiltration_depth_in_mm"
]
ORDINAL_FEATURES = [
    "pT_stage",
    "pN_stage",
]
BLOOD_FEATURES = [
    "Leukocytes [#/volume] in Blood",
    "Hemoglobin [Mass/volume] in Blood",
    "Platelets [#/volume] in Blood",
    "Erythrocytes [#/volume] in Blood",
    "Hematocrit [Volume Fraction] of Blood",
    "Erythrocyte mean corpuscular hemoglobin [Entitic mass]",
    "Erythrocyte mean corpuscular volume [Entitic volume]",
    "Erythrocyte mean corpuscular hemoglobin concentration [Mass/volume]",
    "Erythrocyte distribution width [Ratio]",
    "Platelet mean volume [Entitic volume] in Blood",
    "Granulocytes [#/volume] in Blood",
    "Eosinophils [#/volume] in Blood",
    "Basophils [#/volume] in Blood",
    "Lymphocytes [#/volume] in Blood",
    "Monocytes [#/volume] in Blood",
    "Platelet distribution width [Entitic volume] in Blood by Automated count"
]


def get_binary_features(df_original, verbose=0):
    if verbose == 1:
        print("\n=== Binary features ===")
    df = pd.DataFrame()
    binary_features = [col for col in df_original.columns if col in BINARY_FEATURES]
    df[binary_features] = df_original[binary_features]

    for col in binary_features:
        # strings to binary
        df[col] = df[col].replace({"alive": 0, "dead": 1,
                                   "no": 0, "yes": 1,
                                   "Absent": 0, "CIS": 1,
                                   "male": 0, "female": 1})
        if verbose == 1:
            print(col, df[col].unique())

    return df


def get_nominal_features(df_original, verbose=0):
    if verbose == 1:
        print("\n=== Nominal features ===")
    df = pd.DataFrame()
    nominal_features = [col for col in df_original.columns if col in NOMINAL_FEATURES]

    for feature in nominal_features:
        # Get values of current feature
        temp = df_original[feature]
        # Label encoding
        labelencoder = LabelEncoder()
        encoded = labelencoder.fit_transform(temp)
        df[feature] = encoded
        le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
        if verbose == 1:
            print(f"{feature} {le_name_mapping}\n")

    return df


def get_discrete_features(df_original, verbose=0):
    df = pd.DataFrame()
    discrete_features = [col for col in df_original.columns if col in DISCRETE_FEATURES]

    df[discrete_features] = df_original[discrete_features]

    if verbose == 1:
        print("\n=== Discrete features ===")
        for feature in discrete_features:
            print(f"{feature}: mean = {df[feature].mean():.2f}, var = {df[feature].std():.2f}")
    return df


def get_ordinal_features(df_original, verbose=0):
    df_copy = df_original.copy()
    ordinal_features = [col for col in df_original.columns if col in ORDINAL_FEATURES]

    if verbose == 1:
        print("\n=== Ordinal features ===")

    df = pd.DataFrame()
    for feature in ordinal_features:
        if feature == "pT_stage":
            # Change order to [pTis, pTX, pT1, pT2, ...]
            df_copy["pT_stage"] = df_copy["pT_stage"].replace({"pTis": "T0is"})  # pTis = in situ

        le = LabelEncoder()
        df[feature] = le.fit_transform(df_copy[feature])
        if verbose == 1:
            print(feature, list(le.classes_))
    return df


def get_tabular_features(file_path, verbose=0):
    df = pd.read_json(file_path, dtype={"patient_id": str})
    df_binary = get_binary_features(df, verbose)
    df_nominal = get_nominal_features(df, verbose)
    df_discrete = get_discrete_features(df, verbose)
    df_ordinal = get_ordinal_features(df, verbose)

    features_df = pd.concat([df_binary, df_nominal, df_discrete, df_ordinal], axis=1)
    features = features_df.to_numpy()
    features_df.insert(0, "patient_id", df["patient_id"])
    return features, features_df


def get_mode(df_blood, normal, patient_ids, verbose=0):
    """
    Plot histograms and calculate mode for each parameter.

    :param df_blood: DataFrame
    :param normal: DataFrame with normal ranges
    :param patient_ids: List with patient_ids
    :return: Dictionary {loinc : mode}
    """
    df = df_blood[df_blood.patient_id.isin(patient_ids)]

    fig, axs = plt.subplots(4, 4, figsize=(10, 6))
    axs = axs.ravel()

    mode_dict = {}

    for i, loinc in enumerate(df.columns[1:]):
        y = axs[i].hist(df[loinc], bins=30)

        n, bins = y[0], y[1]
        idx_max = np.argmax(n)
        mode = (bins[idx_max] + bins[idx_max + 1]) / 2
        mode_dict[loinc] = mode

        axs[i].axvline(mode, color="orange", label="mode")
        param = normal[normal.LOINC_name == loinc].analyte_name.values[0]
        param += " [" + normal[normal.LOINC_name == loinc].unit.values[0] + "]"
        axs[i].set_xlabel(param)

    if verbose == 1:
        plt.tight_layout()
        plt.legend()
        sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
        plt.show()
    else:
        plt.close()

    return mode_dict


def fill_missing_values(df, fill_dict_f, fill_dict_m, ids_f, ids_m, verbose=0):
    """
    Fill missing values (NaNs) with values from dictionaries, separately for males and feamales
    :param df: original Dataframe
    :param fill_dict_f: Dictionary {loinc: mode} for females
    :param fill_dict_m: Dictionary {loinc: mode} for males
    :param ids_f: patient IDs of females
    :param ids_m: patinet IDs of males
    :return: DataFrame without missing values
    """
    idx_f = df[df.patient_id.isin(ids_f)].index
    idx_m = df[df.patient_id.isin(ids_m)].index
    if verbose == 1:
        print("Missing values:", df.isnull().sum().sum())
    for loinc in df.columns[1:]:
        df.loc[idx_f, loinc] = df.loc[idx_f, loinc].fillna(fill_dict_f[loinc])
        df.loc[idx_m, loinc] = df.loc[idx_m, loinc].fillna(fill_dict_m[loinc])
    if verbose == 1:
        print("Missing values after replacing:", df.isnull().sum().sum())
    return df


def get_blood_features(file_path_blood, file_path_normal, file_path_clinical, impute_missing=True, verbose=0):
    data = pd.read_json(file_path_blood, dtype={"patient_id": str})
    clinical = pd.read_json(file_path_clinical, dtype={"patient_id": str})
    ref = pd.read_json(file_path_normal)

    data = data.pivot_table(index="patient_id", columns="LOINC_name", values="value", aggfunc="first")
    data = data.reset_index().rename_axis(None, axis=1)

    # Select parameters
    parameters = list(ref[(ref.group == "Hematology") & (ref.normal_female_max.notnull())].LOINC_name)
    parameters = [p for p in parameters if not p[-1] == "%"]
    data = data[["patient_id"] + parameters]

    # Drop outlier
    cols = ["patient_id", "Basophils [#/volume] in Blood", "Leukocytes [#/volume] in Blood"]
    subset = data[cols].copy()
    outlier_idx = subset[subset.columns[1:]].idxmax().unique()
    assert len(outlier_idx) == 1, "Patient with outlier values not found."
    subset["outlier"] = ["no"]*len(subset)
    subset.iloc[outlier_idx, -1] = "yes"
    if verbose == 1:
        fig, axs = plt.subplots(2, figsize=(4, 2))
        for i in range(2):
            sns.stripplot(subset, x=subset.columns[i + 1], s=3, jitter=0.3, ax=axs[i], hue="outlier")
            sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
        plt.title("Outlier to be removed")
        plt.tight_layout()
        plt.show()
    data = data.drop(outlier_idx, axis=0).reset_index(drop=True)

    if impute_missing:
        # Get patient patient_ids for males and females
        ids_male = clinical[clinical.sex == "male"].patient_id.tolist()
        ids_female = clinical[clinical.sex == "female"].patient_id.tolist()

        # Calculate mode from histogram of each parameter
        modes_male = get_mode(data, ref, ids_male, verbose)
        modes_female = get_mode(data, ref, ids_female, verbose)
        data = fill_missing_values(data, modes_female, modes_male, ids_female, ids_male, verbose)

    # Save raw values
    features = data[data.columns[1:]].to_numpy()
    return features, data


def get_target_classes(file_path_clinical):
    cols = [
        "patient_id", "recurrence", "days_to_recurrence",
        "survival_status", "survival_status_with_cause", "days_to_last_information",
        "rfs_event", "days_to_rfs_event"
    ]
    rfs_cols = ["days_to_recurrence", "days_to_metastasis_1", "days_to_progress_1", "days_to_death"]

    df = pd.read_json(file_path_clinical, dtype={"patient_id": str})
    df["days_to_death"] = df.apply(
        lambda x: x["days_to_last_information"] if x["survival_status"] == "deceased" else None, axis=1)
    df["days_to_rfs_event"] = df.apply(lambda x: x[rfs_cols].min(skipna=True) - x["days_to_first_treatment"], axis=1)
    df["rfs_event"] = df.apply(lambda x: 0 if pd.isnull(x["days_to_rfs_event"]) else 1, axis=1)
    df["days_to_rfs_event"] = df["days_to_rfs_event"].fillna(df["days_to_last_information"] - df["days_to_first_treatment"])
    df["days_to_rfs_event"] = df["days_to_rfs_event"].astype(int)

    return df[cols]
