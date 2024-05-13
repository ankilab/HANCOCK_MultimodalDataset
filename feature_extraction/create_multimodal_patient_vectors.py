from extract_text_features import get_icd_vectors
from extract_tabular_features import get_tabular_features, get_blood_features, get_target_classes
from extract_tma_features import get_tma_features
import numpy as np
from argparse import ArgumentParser
from pathlib import Path


def save_numpy_compressed(patient_ids, features, file_path):
    """
    Save vectors with patient IDs as keys to a numpy compressed file (.npz)

    Parameters
    ----------
    patient_ids : list of strings
        List of patient IDs
    features : numpy array
        Features
    file_path : string
        File path for saving the file
    """
    savez_dict = {}
    for i in range(features.shape[0]):
        patient_id = patient_ids[i]
        savez_dict[str(int(patient_id))] = features[i, :]
    np.savez_compressed(file_path, **savez_dict)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir_icd_codes", type=str, help="Path to directory with ICD code text files")
    parser.add_argument("dir_structureddata", type=str, help="Path to directory containing tabular json files")
    parser.add_argument("path_tma_measurements", type=str, help="Path to 'TMA_celldensity_measurements.csv'")
    parser.add_argument("features_dir", type=str, help="Path to directory where the features are saved")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show additional information")
    parser.add_argument("--npz", action="store_true", help="Save features to compressed numpy file instead of csv")
    args = parser.parse_args()

    dest_dir = Path(args.features_dir)
    json_dir = Path(args.dir_structureddata)

    # Save text features (ICD codes as bag of words)
    icd_vectors, icd_df, _ = get_icd_vectors(args.dir_icd_codes)
    if args.npz:
        save_numpy_compressed(icd_df.patient_id.tolist(), icd_vectors, str(dest_dir / "icd_codes"))
    else:
        icd_df.to_csv(dest_dir/"icd_codes.csv", index=False)

    # Save clinicial features
    clinical_vectors, clinical_df = get_tabular_features(json_dir/"clinical_data.json", verbose=args.verbose)
    if args.npz:
        save_numpy_compressed(clinical_df.patient_id.tolist(), clinical_vectors, str(dest_dir/"clinical"))
    else:
        clinical_df.to_csv(dest_dir/"clinical.csv", index=False)

    # Save pathological features
    patho_vectors, patho_df = get_tabular_features(json_dir/"pathological_data.json", verbose=args.verbose)
    if args.npz:
        save_numpy_compressed(patho_df.patient_id.tolist(), patho_vectors, str(dest_dir/"pathological"))
    else:
        patho_df.to_csv(dest_dir/"pathological.csv", index=False)

    # Save blood parameters
    blood_vectors, blood_df = get_blood_features(
        file_path_blood=json_dir/"blood_data.json",
        file_path_normal=json_dir/"blood_data_reference_ranges.json",
        file_path_clinical=json_dir/"clinical_data.json",
        verbose=args.verbose
    )
    if args.npz:
        save_numpy_compressed(blood_df.patient_id.tolist(), blood_vectors, str(dest_dir/"blood"))
    else:
        blood_df.to_csv(dest_dir / "blood.csv", index=False)

    # Save cell densities from CD3 and CD8 TMAs
    tma_vectors, tma_df = get_tma_features(args.path_tma_measurements)
    if args.npz:
        save_numpy_compressed(tma_df.patient_id.tolist(), tma_vectors, str(dest_dir / "tma_cell_density"))
    else:
        tma_df.to_csv(dest_dir/"tma_cell_density.csv", index=False)

    # Save target classes (recurrence and survival)
    target_df = get_target_classes(json_dir/"clinical_data.json")
    target_df.to_csv(dest_dir/"targets.csv", index=False)
