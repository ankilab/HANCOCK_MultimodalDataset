import pandas as pd


TMA_FEATURES = ["cd3_z", "cd3_inv", "cd8_z", "cd8_inv"]


def get_tma_features(tma_measurement_file):
    df = pd.read_csv(tma_measurement_file, dtype={"Case ID": str})
    
    # drop missing cores
    missing_idx = df[df["Missing"] == True].index
    df = df.drop(missing_idx)
    df["Missing"].value_counts()

    # drop cores with no Case ID
    nan_idx = df[df["Case ID"].isna()].index
    df = df.drop(nan_idx)

    # Extract location from filename
    df["location"] = df["Image"].str.extract(r"(TumorCenter|InvasionFront)")
    df["marker"] = df["Image"].str.extract(r"(CD3|CD8)")

    cd3 = df[df.marker == "CD3"]
    cd8 = df[df.marker == "CD8"]

    # Change dataframe structure
    cols = ["Positive %", "Num Positive per mm^2", "location", "Case ID"]
    cd8_mean = cd8[cols].groupby(["Case ID", "location"]).mean().reset_index()
    cd3_mean = cd3[cols].groupby(["Case ID", "location"]).mean().reset_index()

    # Split by location (z = tumor center, inv = invasion front)
    cd3_mean_z = cd3_mean[cd3_mean["location"] == "TumorCenter"].reset_index()
    cd3_mean_inv = cd3_mean[cd3_mean["location"] == "InvasionFront"].reset_index()
    cd8_mean_z = cd8_mean[cd8_mean["location"] == "TumorCenter"].reset_index()
    cd8_mean_inv = cd8_mean[cd8_mean["location"] == "InvasionFront"].reset_index()

    # Select and rename columns
    cols = ["Case ID", "Num Positive per mm^2"]
    cd3_mean_z = cd3_mean_z[cols].rename(columns={cols[1]: "cd3_z"})
    cd3_mean_inv = cd3_mean_inv[cols].rename(columns={cols[1]: "cd3_inv"})
    cd8_mean_z = cd8_mean_z[cols].rename(columns={cols[1]: "cd8_z"})
    cd8_mean_inv = cd8_mean_inv[cols].rename(columns={cols[1]: "cd8_inv"})

    # Merge to one dataframe
    cd3_cd8 = pd.merge(cd3_mean_z, cd3_mean_inv, on="Case ID", how="outer")
    cd3_cd8 = pd.merge(cd3_cd8, cd8_mean_z, on="Case ID", how="outer")
    cd3_cd8 = pd.merge(cd3_cd8, cd8_mean_inv, on="Case ID", how="outer")
    cd3_cd8 = cd3_cd8.reset_index(drop=True)
    cd3_cd8 = cd3_cd8.sort_values(by="Case ID").rename(columns={"Case ID": "patient_id"})

    features = cd3_cd8[cd3_cd8.columns[1:]].to_numpy()

    return features, cd3_cd8
