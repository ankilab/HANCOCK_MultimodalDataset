from argparse import ArgumentParser
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from lifelines import KaplanMeierFitter, statistics
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.utils import compute_class_weight

from keras.utils import Sequence
from keras import layers
from keras.models import Model
from keras import optimizers
from keras.callbacks import CSVLogger, TensorBoard
from keras.metrics import Precision, Recall
from keras.utils import set_random_seed
from keras.backend import clear_session

from data_exploration.umap_embedding import setup_preprocessing_pipeline
from utils import get_significance


SEED = 42


def ConvBlock(x, filters, norm):
    # Conv -> (Norm) -> ReLu
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    if norm == "batch":
        x = layers.BatchNormalization()(x)
   # elif norm == "instance":
   #     x = InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    # Conv -> (Norm) -> ReLu
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    if norm == "batch":
        x = layers.BatchNormalization()(x)
    #elif norm == "instance":
    #    x = InstanceNormalization()(x)
    x = layers.Activation("relu")(x)

    # Pooling
    x = layers.MaxPooling2D((2, 2))(x)

    return x


def ConvNet(in_shape, num_conv_blocks, initial_filters, dense_units, dropout, norm, constant_filters):
    """
    Convolutional Neural Network

    Parameters
    ----------
    in_shape : tuple
        Input shape
    num_conv_blocks : int
        Number of convolutional blocks, each has 2 Conv2D layers and a MaxPooling layer
    initial_filters : int
        Initial number of convolutional filters, for the i-th layer it is multiplied with 2**i
    dense_units : int
        Number of units in the dense layer
    dropout : bool
        Whether to use Dropout or not (p=0.3)
    norm : string
        Set to 'batch' or 'instance', default is no normalization layer
    constant_filters : bool
        Same number of filters in every Conv2D layer if True, increasing number if false

    Returns
        Keras Functional Model
    -------

    """
    inputs = layers.Input(shape=in_shape)
    x = inputs

    for i in range(num_conv_blocks):
        factor = 1 if constant_filters else (2 ** i)
        x = ConvBlock(x, filters=initial_filters * factor, norm=norm)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)

    if dropout:
        x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)


class DataGenerator2D(Sequence):
    def __init__(self, data_array, labels, batch_size, output_shape, shuffle,
                 min_max_scaling=True, tiles_per_patient=48,
                 min_value=None, max_value=None,
                 set_patient_vectors_to_zero=False
                 ):
        self.data = data_array
        self.labels = labels
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.shuffle = shuffle
        self.min_max_scaling = min_max_scaling
        self.min_value = min_value
        self.max_value = max_value
        if self.min_max_scaling:
            assert (self.min_value is not None) and (self.max_value is not None), "If min_max_scaling is applied, min_value and max_value must be set."
        else:
            if (self.min_value is not None) or (self.max_value is not None):
                print("Warning: Min or max value were given, but min_max_scaling is not applied because it is set to False.")
        self.tiles_per_patient = tiles_per_patient
        self.set_patient_vectors_to_zero = set_patient_vectors_to_zero
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels)/self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self._generate_batch(batch_indexes)
        return X, y

    def _generate_batch(self, batch_indexes):
        # Labels
        y = np.ones(self.batch_size, dtype="float32") * (-1)
        X = np.ones((self.batch_size, *self.output_shape), dtype="float32")

        num_markers = 8
        dim = 256

        for i, batch_idx in enumerate(batch_indexes):
            # Get one row (data of one patient)
            patient_data = self.data[batch_idx]

            non_image_emb = patient_data[:-self.tiles_per_patient * num_markers * dim]
            image_emb = patient_data[-self.tiles_per_patient * num_markers * dim:]

            if self.set_patient_vectors_to_zero:
                non_image_emb = np.zeros(non_image_emb.shape)

            if self.min_max_scaling:
                image_emb = (image_emb - self.min_value)/(self.max_value - self.min_value + 1e-6)

            # Class label
            class_label = self.labels[batch_idx]

            # Pad array with zeros at the end
            target_length = self.output_shape[0] * self.output_shape[1]
            pad = target_length - len(patient_data)
            patient_data = np.concatenate([image_emb, non_image_emb, np.zeros(pad)])

            # Reshape 1D array to 2D
            patient_data = patient_data.reshape((self.output_shape[0], self.output_shape[1]))

            # Add sample to batch
            X[i] = patient_data[..., None]
            y[i] = class_label

        return X, y


def expand_vector(row, column_name):
    return pd.Series(row[column_name])


def cross_validation(dataframe, k=10):
    auc_list = []
    tpr_list = []
    x_linspace = np.linspace(0, 1, 100)
    fold_idx = 1

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for train_idx, val_idx in cv.split(df_train, df_train["target"]):
        print(f"\n **** Fold {fold_idx}\n")

        # Split folds
        df_train_folds = dataframe.iloc[train_idx]
        df_val_fold = dataframe.iloc[val_idx]

        # Preprocess data
        preprocessor = setup_preprocessing_pipeline(df_train_folds.columns[2:])
        y_train = df_train_folds["target"].to_numpy()
        y_val = df_val_fold["target"].to_numpy()
        X_train = preprocessor.fit_transform(df_train_folds.drop(["patient_id", "target"], axis=1))
        X_val = preprocessor.transform(df_val_fold.drop(["patient_id", "target"], axis=1))

        # Compute min and max of all image embeddings in the training set
        dims = params["tiles_per_patient"] * params["dtr_embedding_dim"] * len(markers)
        min_value = np.min(X_train[:, -dims:], axis=(0, 1))
        max_value = np.max(X_train[:, -dims:], axis=(0, 1))

        # Set up data generators
        train_gen = DataGenerator2D(X_train, y_train, batch_size=16, output_shape=params["input_shape"], shuffle=True,
                                    tiles_per_patient=2, min_max_scaling=True, min_value=min_value, max_value=max_value,
                                    set_patient_vectors_to_zero=params["exclude_multimodal_patient_vectors"])
        val_gen = DataGenerator2D(X_val, y_val, batch_size=1, output_shape=params["input_shape"], shuffle=False,
                                  tiles_per_patient=2, min_max_scaling=True, min_value=min_value, max_value=max_value,
                                  set_patient_vectors_to_zero=params["exclude_multimodal_patient_vectors"])

        # Build Convolutional Neural Network
        model = ConvNet(
            in_shape=params["input_shape"],
            num_conv_blocks=params["num_conv_blocks"],
            initial_filters=params["initial_filters"],
            constant_filters=params["constant_filters"],
            dense_units=params["dense_units"],
            norm=params["norm"],
            dropout=params["dropout"]
        )
        if fold_idx == 1:
            print(model.summary())
        model.compile(
            optimizer=optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy", Precision(), Recall()]
        )
        logger = CSVLogger(filename=f"{args.results_directory}/{args.prefix}convnet_history_fold{fold_idx}.csv")
        tensorboard = TensorBoard(log_dir=f"{args.results_directory}/tensorboard_logs/{args.prefix}convnet_fold{fold_idx}")
        callbacks = [logger, tensorboard] if args.tensorboard else [logger]

        # Class weights
        class_weight = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight = {0: class_weight[0], 1: class_weight[1]}

        # Training
        model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            validation_data=val_gen,
            epochs=params["epochs"],
            callbacks=callbacks,
            class_weight=class_weight
        )

        # Get predictions
        y_pred = model.predict(val_gen)

        # Store values for ROC curve
        fpr, tpr, thresh = roc_curve(y_val, y_pred)
        tpr = np.interp(x_linspace, fpr, tpr)
        tpr[0] = 0.0
        tpr[-1] = 1.0
        tpr_list.append(tpr)
        auc_list.append(roc_auc_score(y_val, y_pred))

        fold_idx += 1
        clear_session()

    return tpr_list, auc_list


def training_and_testing(dataframe, dataframe_test):
    # Preprocess data
    preprocessor = setup_preprocessing_pipeline(dataframe.columns[2:])
    y_train = dataframe["target"].to_numpy()
    y_test = dataframe_test["target"].to_numpy()
    X_train = preprocessor.fit_transform(dataframe.drop(["patient_id", "target"], axis=1))
    X_test = preprocessor.transform(dataframe_test.drop(["patient_id", "target"], axis=1))

    # Compute min and max of all image embeddings in the training set
    dims = params["tiles_per_patient"] * params["dtr_embedding_dim"] * len(markers)
    min_value = np.min(X_train[:, -dims:], axis=(0, 1))
    max_value = np.max(X_train[:, -dims:], axis=(0, 1))

    # Set up data generators
    train_gen = DataGenerator2D(X_train, y_train, batch_size=16, output_shape=params["input_shape"], shuffle=True,
                                tiles_per_patient=2, min_max_scaling=True, min_value=min_value, max_value=max_value,
                                set_patient_vectors_to_zero=params["exclude_multimodal_patient_vectors"])
    test_gen = DataGenerator2D(X_test, y_test, batch_size=1, output_shape=params["input_shape"], shuffle=False,
                               tiles_per_patient=2, min_max_scaling=True, min_value=min_value, max_value=max_value,
                               set_patient_vectors_to_zero=params["exclude_multimodal_patient_vectors"])

    # Build Convolutional Neural Network
    model = ConvNet(
        in_shape=params["input_shape"],
        num_conv_blocks=params["num_conv_blocks"],
        initial_filters=params["initial_filters"],
        constant_filters=params["constant_filters"],
        dense_units=params["dense_units"],
        norm=params["norm"],
        dropout=params["dropout"]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )
    # Class weights
    class_weight = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {0: class_weight[0], 1: class_weight[1]}

    # Training
    model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=params["epochs"],
        class_weight=class_weight
    )

    # Get predictions
    y_pred = model.predict(test_gen)[:, 0]

    print(classification_report(y_test, y_pred > 0.5))

    # Threshold predictions
    y_pred = (y_pred > 0.5).astype(int)

    # Load survival data for test cases
    os_rfs = pd.read_csv(data_dir/"targets.csv", dtype={"patient_id": str})
    os_rfs = os_rfs[os_rfs.patient_id.isin(dataframe_test.patient_id.tolist())].reset_index(drop=True)
    # Define event: 1 = deceased, 0 = censored
    os_rfs.survival_status = os_rfs.survival_status == "deceased"
    os_rfs.survival_status = os_rfs.survival_status.astype(int)
    # Define event: 1 = recurrence, 0 = censored
    os_rfs.recurrence = os_rfs.recurrence == "yes"
    os_rfs.recurrence = os_rfs.recurrence.astype(int)
    # Convert number of days to number of months
    avg_days_per_month = 365.25/12
    os_rfs["followup_months"] = np.round(os_rfs["days_to_last_information"]/avg_days_per_month)
    os_rfs["months_to_rfs_event"] = np.round(os_rfs["days_to_rfs_event"]/avg_days_per_month)

    # Split cases by predictions
    idx0 = np.argwhere(y_pred == 0).flatten()
    idx1 = np.argwhere(y_pred == 1).flatten()
    pred_0 = os_rfs.iloc[idx0].reset_index(drop=True)
    pred_1 = os_rfs.iloc[idx1].reset_index(drop=True)

    # Log-rank test (survival)
    logrank_result_os = statistics.logrank_test(
        durations_A=pred_0.followup_months,
        durations_B=pred_1.followup_months,
        event_observed_A=pred_0.survival_status,
        event_observed_B=pred_1.survival_status
    )
    p_os = logrank_result_os.p_value
    p_value_os = get_significance(p_os)

    # Overall survival, grouped by predictions
    plt.figure(figsize=(2, 1.4))

    # Kaplan-Meier estimate
    kmf0 = KaplanMeierFitter()
    kmf0.fit(pred_0.followup_months, pred_0.survival_status, label="pred. no adjuvant therapy")
    ax = kmf0.plot_survival_function(ci_show=True, show_censors=False, color=sns.color_palette("Set2")[0], linewidth=1)

    kmf1 = KaplanMeierFitter()
    kmf1.fit(pred_1.followup_months, pred_1.survival_status, label="pred. adjuvant therapy")
    kmf1.plot_survival_function(ci_show=True, show_censors=False, color=sns.color_palette("Set2")[1], linewidth=1)

    plt.text(50, 0.9, p_value_os)
    plt.ylabel("Overall survival", fontsize=6)
    plt.xlabel("Time since diagnosis [months]", fontsize=6)
    plt.xlim([0, ax.get_xticks()[-2]])
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.5))
    plt.savefig(results_dir/f"{args.prefix}convnet_adjuvant_treatment_prediction_os.svg", bbox_inches="tight")
    plt.close()

    # Log-rank test (Recurrence-free survival, grouped by predictions)
    logrank_result_os = statistics.logrank_test(
        durations_A=pred_0.months_to_rfs_event,
        durations_B=pred_1.months_to_rfs_event,
        event_observed_A=pred_0.rfs_event,
        event_observed_B=pred_1.rfs_event
    )
    p_rfs = logrank_result_os.p_value
    p_value_rfs = get_significance(p_rfs)

    plt.figure(figsize=(2, 1.4))

    kmf0 = KaplanMeierFitter()
    kmf0.fit(pred_0.months_to_rfs_event, pred_0.rfs_event, label="pred. no adjuvant therapy")
    ax = kmf0.plot_survival_function(ci_show=True, show_censors=False, color=sns.color_palette("Set2")[0], linewidth=1)

    kmf1 = KaplanMeierFitter()
    kmf1.fit(pred_1.months_to_rfs_event, pred_1.rfs_event, label="pred. adjuvant therapy")
    kmf1.plot_survival_function(ci_show=True, show_censors=False, color=sns.color_palette("Set2")[1], linewidth=1)

    plt.text(50, 0.9, p_value_rfs)
    plt.ylabel("Recurrence-free survival", fontsize=6)
    plt.xlabel("Time since surgery [months]", fontsize=6)
    plt.xlim([0, ax.get_xticks()[-2]])
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.5))
    plt.savefig(results_dir/f"{args.prefix}convnet_adjuvant_treatment_prediction_rfs.svg", bbox_inches="tight")
    plt.close()

    # Bar plot showing predictions for test dataset
    plt.figure(figsize=(2.2, 1))
    num_pred_adjuvant = np.count_nonzero(y_pred)
    num_pred_none = np.count_nonzero(y_pred == 0)
    ax = plt.barh([0, 1], [num_pred_none, num_pred_adjuvant], color=sns.color_palette("Set2")[1])
    plt.bar_label(ax, fontsize=6, padding=2)
    plt.yticks(ticks=[0, 1], labels=["No adjuvant therapy", "Adjuvant therapy"])
    plt.xticks([0, 20, 40, 60, 80])
    plt.ylabel("Prediction", fontsize=6)
    plt.xlabel("# Patients", fontsize=6)
    plt.tight_layout()
    sns.despine()
    plt.savefig(results_dir/f"{args.prefix}convnet_adjuvant_treatment_prediction_barplot.svg", bbox_inches="tight")
    plt.close()

    # Explaining test predictions using SHAP values
    # Get 100 random training batches
    train_gen.on_epoch_end()  # shuffle
    i = 0
    background = []
    for x, y in train_gen:
        if i >= 100:
            break
        background.append(x)
        i += 1
    background = np.concatenate(background)

    # Set up SHAP Explainer
    explainer = shap.DeepExplainer(model, background)

    # Save image plots for all test samples
    print("Saving SHAP image plots...")
    if not os.path.exists(results_dir/f"{args.prefix}shap_image_plots"):
        os.mkdir(results_dir/f"{args.prefix}shap_image_plots")
    i = 0
    test_gen.on_epoch_end()
    for x, y in tqdm(test_gen):
        shap_values = explainer.shap_values(x)
        plt.figure(figsize=(5, 3))
        shap.image_plot(shap_values, -x, show=False)
        pred = model.predict(x, verbose=0)
        plt.title(f"Class: {y[0]:.0f}, predicted: {pred[0][0]:.3f}")
        plt.savefig(results_dir/f"{args.prefix}shap_image_plots/shap_image_plot_{i}.svg", bbox_inches="tight")
        plt.close()
        i += 1


def visualize_2d_embedding():
    cdim = len(clinical.columns) - 1
    pdim = len(patho.columns) - 1
    bdim = len(blood.columns) - 1
    idim = len(icd.columns) - 1
    ddim = len(cell_density.columns) - 1
    tdim = len(tma_expanded.columns) - 1

    width, height = params["input_shape"][0], params["input_shape"][1]

    modalities_labels = ["tma representations", "clinical", "pathological", "blood", "icd codes", "cell density", "zeros"]
    modalities_2d = [0]*tdim + [1]*cdim + [2]*pdim + [3]*bdim + [4]*idim + [5]*ddim
    empty_dim = width * height - len(modalities_2d)
    modalities_2d += [6] * empty_dim
    modalities_2d = np.array(modalities_2d).reshape((width, height))

    colors = [sns.color_palette("Set2")[i] for i in range(6)] + [sns.color_palette("Set2")[-1]]
    patch_list = [patches.Patch(color=colors[i], label=modalities_labels[i]) for i in range(len(modalities_labels))]
    cmap = ListedColormap(colors)

    plt.figure(figsize=(10, 6))
    plt.imshow(modalities_2d, cmap=cmap)
    plt.legend(handles=patch_list, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasplit_directory", type=str, help="Path to directory that contains data splits as JSON files")
    parser.add_argument("features_directory", type=str, help="Path to directory with extracted features")
    parser.add_argument("results_directory", type=str, help="Path to directory where results will be saved")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--prefix", dest="prefix", type=str, default="", help="Custom prefix for filenames")
    args = parser.parse_args()

    data_dir = Path(args.features_directory)
    results_dir = Path(args.results_directory)

    # Load hyperparameters from json file
    with open("hyperparameters.json", "r") as json_file:
        params = json.load(json_file)

    # Create directory for TensorBoard logging
        # Make folders for logs and plots
        if not os.path.exists(results_dir/"tensorboard_logs"):
            os.mkdir(results_dir/"tensorboard_logs")

    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"

    set_random_seed(SEED)

    # Load extracted features
    clinical = pd.read_csv(data_dir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(data_dir/"pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(data_dir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(data_dir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density = pd.read_csv(data_dir/"tma_cell_density.csv", dtype={"patient_id": str})

    # Load image features
    tma_list = []
    embedding_dim = params["dtr_embedding_dim"]
    markers = ["HE", "CD3", "CD8", "CD56", "CD68", "CD163", "MHC1", "PDL1"]
    for marker in markers:
        file = data_dir/f"centertile_dtr_{embedding_dim}dim_{marker}.npz"
        npz_data = np.load(str(file), allow_pickle=True)
        patient_ids = npz_data.keys()

        data = []
        for pid in patient_ids:
            data.append(npz_data[pid])

        tma_list.append(pd.DataFrame({"patient_id": patient_ids, marker: data}))

    tma = tma_list[0]
    for temp in tma_list[1:]:
        tma = pd.merge(tma, temp, on="patient_id", how="outer")
    tma = tma.fillna(0)

    # Expand vectors to columns
    tma_expanded = pd.DataFrame()
    for marker in markers:
        # workaround: replace scalar 0 wih array of zeros
        tma[marker] = tma.apply(lambda row: np.zeros(512) if np.all(np.asarray(row[marker]) == 0) or np.asarray(row[marker]).ndim != 1 else row[marker], axis=1)
        # Split arrays into df columns
        expanded_cols = tma.apply(lambda row, col=marker: expand_vector(row, col), axis=1)
        column_names = [marker] * params["tiles_per_patient"] * params["dtr_embedding_dim"]
        column_names = [f"{x}_{i}" for x, i in enumerate(column_names)]
        expanded_cols.columns = column_names
        tma_expanded = pd.concat([tma_expanded, expanded_cols], axis=1)

    # Merge all data into a dataframe
    data = clinical.copy()
    if not params["exclude_patho_and_cellcount"]:
        data = data.merge(patho, on="patient_id", how="outer")
        data = data.merge(cell_density, on="patient_id", how="outer")
    data = data.merge(blood, on="patient_id", how="outer")
    data = data.merge(icd, on="patient_id", how="outer")
    data = data.reset_index(drop=True)
    tma_expanded["patient_id"] = tma["patient_id"]
    data = data.merge(tma_expanded, on="patient_id", how="outer")

    # Get targets and train/test split
    target_df = pd.read_json(results_dir/"dataset_split_treatment_outcome.json", dtype={"patient_id": str})
    target_df["target"] = target_df["adjuvant_treatment"].apply(lambda x: 0 if x == "none" else 1)
    target_train = target_df[target_df.dataset == "training"][["patient_id", "target"]]
    target_test = target_df[target_df.dataset == "test"][["patient_id", "target"]]
    df_train = target_train.merge(data, on="patient_id", how="inner")
    df_test = target_test.merge(data, on="patient_id", how="inner")

    # Run 10-fold cross-validation
    true_positive_rates, areas_under_the_curve = cross_validation(df_train)

    # Compute mean and std true positive rates
    mean_tpr = np.mean(true_positive_rates, axis=0)
    std_tpr = np.std(true_positive_rates, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    x_linspace = np.linspace(0, 1, 100)

    # Plot ROC Curve (mean +/- standard deviation) with AUC score
    color = (132/255, 163/255, 204/255)
    plt.figure(figsize=(2.1, 2.3))
    plt.plot(x_linspace, mean_tpr, lw=2, color=color, label=f"AUC = {np.mean(areas_under_the_curve):.2f}")
    plt.fill_between(x_linspace, tpr_lower, tpr_upper, color=color, alpha=0.3, label="$\pm$ std. dev.")
    plt.plot([0, 1], [0, 1], "--", color="black", lw=1)
    plt.xticks(np.arange(0, 1.1, 0.5))
    plt.yticks(np.arange(0, 1.1, 0.5))
    plt.gca().set_aspect("equal")
    plt.xlabel("FPR", fontsize=8)
    plt.ylabel("TPR", fontsize=8)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{args.prefix}convnet_ROC_10fold.svg", bbox_inches="tight")
    plt.close()

    # Save AUC scores
    df = pd.DataFrame({"fold": list(range(len(areas_under_the_curve))), "AUC": areas_under_the_curve})
    df.to_csv(results_dir/f"{args.prefix}convnet_auc_scores.csv", index=False)

    # Train network once on all training data and visualize predictions for test data
    training_and_testing(df_train, df_test)
