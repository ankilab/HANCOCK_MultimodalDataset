from extract_text_features import get_icd_vectors
from extract_tabular_features import get_tabular_features, get_blood_features, get_target_classes
from extract_tma_features import get_tma_features
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
from argument_parser import HancockArgumentParser


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
    parser = HancockArgumentParser(type='feature_extraction')
    args = parser.parse_args()
    dest_dir = Path(args.features_dir)

    # Save text features (ICD codes as bag of words)
    icd_vectors, icd_df, _ = get_icd_vectors(args.path_icd_codes)
    if args.npz:
        save_numpy_compressed(icd_df.patient_id.tolist(),
                              icd_vectors, str(dest_dir / "icd_codes"))
    else:
        icd_df.to_csv(dest_dir/"icd_codes.csv", index=False)

    # Save clinical features
    clinical_vectors, clinical_df = get_tabular_features(
        args.path_clinical, verbose=args.verbose)
    if args.npz:
        save_numpy_compressed(clinical_df.patient_id.tolist(
        ), clinical_vectors, str(dest_dir/"clinical"))
    else:
        clinical_df.to_csv(dest_dir/"clinical.csv", index=False)

    # Save pathological features
    patho_vectors, patho_df = get_tabular_features(
        args.path_patho, verbose=args.verbose)
    if args.npz:
        save_numpy_compressed(patho_df.patient_id.tolist(
        ), patho_vectors, str(dest_dir/"pathological"))
    else:
        patho_df.to_csv(dest_dir/"pathological.csv", index=False)

    # Save blood parameters
    blood_vectors, blood_df = get_blood_features(
        file_path_blood=args.path_blood,
        file_path_normal=args.path_blood_ref,
        file_path_clinical=args.path_clinical,
        verbose=args.verbose
    )
    if args.npz:
        save_numpy_compressed(blood_df.patient_id.tolist(),
                              blood_vectors, str(dest_dir/"blood"))
    else:
        blood_df.to_csv(dest_dir / "blood.csv", index=False)

    # Save cell densities from CD3 and CD8 TMAs
    tma_vectors, tma_df = get_tma_features(args.path_celldensity)
    if args.npz:
        save_numpy_compressed(tma_df.patient_id.tolist(),
                              tma_vectors, str(dest_dir / "tma_cell_density"))
    else:
        tma_df.to_csv(dest_dir/"tma_cell_density.csv", index=False)

    # Save target classes (recurrence and survival)
    target_df = get_target_classes(args.path_clinical)
    target_df.to_csv(dest_dir/"targets.csv", index=False)
