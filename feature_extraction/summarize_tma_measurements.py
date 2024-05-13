from argparse import ArgumentParser
import os
import glob
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("qupath_projects_root_directory", type=str, help="Path to directory containing QuPath projects")
    parser.add_argument("dest_directory", type=str, help="Path to directory for saving the resulting CSV file")
    args = parser.parse_args()

    # Get all tma measurement files
    pattern = Path(args.data_directory).joinpath("*", "tma_measurements", "*.csv")
    files = glob.glob(str(pattern))

    # Combine into a single file and save it
    df_list = []
    for file in files:
        df_list.append(pd.read_csv(file, sep="\t"))
    df = pd.concat(df_list).drop("Object ID", axis=1)
    df.to_csv(os.path.join(args.dest_directory, "tma_measurements.csv"), index=False)
