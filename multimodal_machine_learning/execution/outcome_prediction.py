from argparse import ArgumentParser
import os
from random import Random

import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline, get_umap_embedding


def return_optimal_random_forest(
        target: str = 'recurrence', random_state: np.random.RandomState = np.random.RandomState(42),
        data_split: str = 'In distribution'
) -> RandomForestClassifier:
    """
    Returns the optimal random forest classifier that is hyperparameter tuned with random search on 100 iterations
    with the evaluation metric F1-Score.

    Args:
        target (str, optional): The target that should be predicted. Available are ['recurrence', 'survival_status'].
            Defaults to 'recurrence'.
        random_state (np.random.RandomState, optional): The random state used for the random forest classifier.
            Defaults to np.random.RandomState(42).
        data_split (str, optional): The type of data split that will be used for determining the test and training split.
            Available are ["In distribution", "Out of distribution", "Oropharynx"]. Defaults to 'In'.

    Raises:
        KeyError: When either the target or the data_split is not supported

        "In distribution"
        "Out of distribution"
        "Oropharynx"
    """
    if target == 'recurrence':
        if data_split == "In distribution":
            return RandomForestClassifier(
                n_estimators=1600, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='log2', max_depth=30,
                criterion='gini',
                random_state=random_state
            )
        elif data_split == "Oropharynx":
            return RandomForestClassifier(
                n_estimators=1200, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='sqrt', max_depth=20,
                criterion='gini',
                random_state=random_state
            )
        elif data_split == "Out of distribution":
            return RandomForestClassifier(
                n_estimators=800, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=100,
                max_features='sqrt', max_depth=80,
                criterion='log_loss',
                random_state=random_state
            )
    elif target == 'survival_status':
        if data_split == "In distribution":
            return RandomForestClassifier(
                n_estimators=1000, min_samples_split=5,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='log2', # max_depth=null,
                criterion='entropy',
                random_state=random_state
            )
        elif data_split == "Oropharynx":
            return RandomForestClassifier(
                n_estimators=1200, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='sqrt', max_depth=20,
                criterion='gini',
                random_state=random_state
            )
        elif data_split == "Out of distribution":
            return RandomForestClassifier(
                n_estimators=1200, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='sqrt', max_depth=20,
                criterion='gini',
                random_state=random_state
            )
    raise KeyError(f'Target {target} or data split {data_split} not recognized')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasplit_directory", type=str, help="Path to directory that contains data splits as JSON files")
    parser.add_argument("features_directory", type=str, help="Path to directory with extracted features")
    parser.add_argument("results_directory", type=str, help="Path to directory where results will be saved")
    parser.add_argument("target", type=str, help="Target class", choices=["recurrence", "survival_status"])
    args = parser.parse_args()

    data_dir = Path(args.features_directory)
    results_dir = Path(args.results_directory)
    split_dir = Path(args.datasplit_directory)
    target = args.target

    # Set seed for reproducibility
    rng = np.random.RandomState(42)

    # Load extracted features
    clinical = pd.read_csv(data_dir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(data_dir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(data_dir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(data_dir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density = pd.read_csv(data_dir/"tma_cell_density.csv", dtype={"patient_id": str})

    # Merge modalities
    df = clinical.merge(patho, on="patient_id", how="outer")
    df = df.merge(blood, on="patient_id", how="outer")
    df = df.merge(icd, on="patient_id", how="outer")
    df = df.merge(cell_density, on="patient_id", how="outer")
    df = df.reset_index(drop=True)

    # Prepare for plots
    x_linspace = np.linspace(0, 1, 100)
    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"
    umap_embeddings = get_umap_embedding(data_dir, umap_min_dist=0.2, umap_n_neighbors=15)

    data_split_paths = [
        split_dir/"dataset_split_in.json",
        split_dir/"dataset_split_out.json",
        split_dir/"dataset_split_Oropharynx.json"
    ]
    data_split_labels = [
        "In distribution",
        "Out of distribution",
        "Oropharynx",
    ]

    tpr_list = [[] for _ in range(len(data_split_paths))]
    auc_list = [[] for _ in range(len(data_split_paths))]

    for i in range(len(data_split_paths)):

        print(f"Training and testing models on {data_split_labels[i]} data...")

        assert_text = f"{data_split_paths[i]} does not exist. Please run genetic_algorithm.py or " \
                      f"split_by_tumor_site.py to generate the corresponding split"
        assert os.path.exists(data_split_paths[i]), assert_text

        # Load patient IDs with dataset split and target classes
        df_split = pd.read_json(data_split_paths[i], dtype={"patient_id": str})[["patient_id", "dataset"]]
        df_targets = pd.read_csv(data_dir/"targets.csv", dtype={"patient_id": str})
        df_split = df_split.merge(df_targets, on="patient_id", how="inner")
        umap_split = umap_embeddings.merge(df_split, on="patient_id", how="inner")

        if target == "recurrence":
            # Only include patients who had a recurrence within 3 years
            # or who survived at least 3 years without recurrence
            df_split = df_split[
                ((df_split.recurrence == "yes") & (df_split.days_to_recurrence <= 365*3)) |
                ((df_split.recurrence == "no") & ((df_split.days_to_last_information > 365*3) |
                                                  (df_split.survival_status == "living")))]
            # Strings to class labels
            df_split.recurrence = df_split.recurrence.replace({"no": 0, "yes": 1})

        elif target == "survival_status":
            # Exclude not tumor specific deaths
            df_split = df_split[~(df_split.survival_status_with_cause == "deceased not tumor specific")]
            # Strings to class labels
            df_split.survival_status = df_split.survival_status.replace({"living": 0, "deceased": 1})

        df_train = df_split[df_split.dataset == "training"][["patient_id", target]].copy()
        df_train.columns = ["patient_id", "target"]
        df_train = df_train.merge(df, on="patient_id", how="inner")

        df_test = df_split[df_split.dataset == "test"][["patient_id", target]].copy()
        df_test.columns = ["patient_id", "target"]
        df_test = df_test.merge(df, on="patient_id", how="inner")

        for iteration in range(5):
            preprocessor = setup_preprocessing_pipeline(df_train.columns[2:])
            y_train = df_train["target"].to_numpy()
            y_test = df_test["target"].to_numpy()
            X_train = preprocessor.fit_transform(df_train.drop(["patient_id", "target"], axis=1))
            X_test = preprocessor.transform(df_test.drop(["patient_id", "target"], axis=1))

            # Handle class imbalance
            smote = SMOTE(random_state=rng)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # Fit ML model
            model = return_optimal_random_forest(target=target, data_split=data_split_labels[i])
            model.fit(X_train, y_train)

            # Get predictions for test dataset
            y_test_predicted = model.predict_proba(X_test)[:, 1]

            # ROC curve
            fpr, tpr, thresh = roc_curve(y_test, y_test_predicted)
            tpr = np.interp(x_linspace, fpr, tpr)
            tpr[0] = 0.0
            tpr[-1] = 1.0
            tpr_list[i].append(tpr)
            auc_list[i].append(roc_auc_score(y_test, y_test_predicted))

            # Plot dataset split in 2D
            plt.figure(figsize=(1.75, 1.75))
            palette = {"training": "lightgrey", "test": sns.color_palette("Set2")[i]}
            ax = sns.scatterplot(umap_split[umap_split.dataset=="training"], x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
            ax = sns.scatterplot(umap_split[umap_split.dataset=="test"], x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
            plt.title(data_split_labels[i])
            ax.set_aspect("equal")
            plt.legend()
            sns.despine()
            plt.xticks([])
            plt.yticks([])
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False, fontsize=6)
            plt.xlabel("UMAP 1", fontsize=6)
            plt.ylabel("UMAP 2", fontsize=6)
            plt.tight_layout()
            plt.savefig(results_dir/f"umap_split_{data_split_labels[i]}.svg", bbox_inches="tight")
            plt.close()

    # Plot ROC curve
    colors = sns.color_palette("Set2")
    plt.figure(figsize=(2.2, 1.75))
    for i in range(len(auc_list)):
        mean_tpr = np.mean(tpr_list[i], axis=0)
        std_tpr = np.std(tpr_list[i], axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_fpr = np.linspace(0, 1, 100)

        plt.plot(x_linspace, mean_tpr, linewidth=0.8, color=colors[i],
                 label=f"AUC = {np.mean(auc_list[i]):.2f}$\pm${np.std(auc_list[i]):.2f}")
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[i], alpha=0.5, lw=0)

    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Random")
    plt.xticks(np.arange(0, 1.2, 0.5))
    plt.yticks(np.arange(0, 1.2, 0.5))
    plt.xlabel("FPR", fontsize=6)
    plt.ylabel("TPR", fontsize=6)
    plt.title(f"{target}")
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(results_dir/f"roc_testsets_{target}.svg", bbox_inches="tight")
    plt.close()

    print(f"Done. Saved results to {results_dir}")
