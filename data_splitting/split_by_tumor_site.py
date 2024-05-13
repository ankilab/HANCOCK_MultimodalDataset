from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


def split_data_by_site(json_dir, results_dir, site):
    """
    Function for splitting data into training and test set by primary tumor site.
    For example, if site is set to "Oropharynx", all cases with oropharyngeal carcinoma are assigned to test,
    and the remaining cases to training.

    Parameters
    ----------
    json_dir : str
        Path to directory containing extracted features (json files)
    results_dir : str
        Path to directory for saving results
    site : str
        Primary tumor site, possible choices are 'Hypopharynx', 'Oropharynx', 'Oral_Cavity', 'Larynx'
    """
    json_dir = Path(json_dir)
    results_dir = Path(results_dir)

    # Split data: All cases with specified site are assigned to test, remaining cases to training data
    df = pd.read_json(json_dir / "pathological_data.json", dtype={"patient_id": str})
    df = df.merge(pd.read_json(json_dir / "clinical_data.json", dtype={"patient_id": str}), on="patient_id",
                  how="inner")
    df["dataset"] = df.apply(lambda x: "test" if x.primary_tumor_site == args.site else "training", axis=1)
    df = df[
        ["patient_id", "dataset"]]

    # Save dataset split as CSV file
    df.to_json(results_dir / f"dataset_split_{site}.json", orient="records", indent=1)

    print(f"#Cases in the test dataset with site {site}:", len(df[df.dataset == "test"]))
    print(f"#Cases in the training dataset:", len(df[df.dataset == "training"]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_directory", type=str, help="Path to directory containing tabular json files")
    parser.add_argument("log_directory", type=str, help="Path to directory where resulting file will be saved")
    parser.add_argument(
        "-s", "--site", dest="site", type=str,
        choices=["Hypopharynx", "Oropharynx", "Oral_Cavity", "Larynx"], default="Larynx",
        help="Primary tumor site that will be assigned to the test dataset"
    )
    args = parser.parse_args()

    split_data_by_site(args.json_directory, args.log_directory, args.site)
