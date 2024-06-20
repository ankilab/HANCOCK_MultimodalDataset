import unittest
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from data_reader_factory import DataFrameReaderFactory
import pandas as pd
import numpy as np


class TestDataReader(unittest.TestCase):
    def __init__(self):
        super().__init__()
        data_root_dir = Path(__file__).parents[2] / 'Hancock_Dataset'
        # self.results_dir = Path(__file__).parents[1] / 'results'
        self.path_clinical = data_root_dir / 'StructuredData' / 'clinical_data.json'
        self.path_patho = data_root_dir / 'StructuredData' / 'pathological_data.json'
        self.path_blood = data_root_dir / 'StructuredData' / 'blood_data.json'

    def test_blood_data_frame_reader(self):
        blood_reader = DataFrameReaderFactory(
        ).make_data_frame_reader('Blood', self.path_blood)
        blood_reader_data = blood_reader.return_data()
        blood_reader_data = blood_reader_data['patient_id'].value_counts(
        ).reset_index()
        blood_reader_data.columns = ['patient_id', 'Blood data']
        blood = pd.read_json(self.path_blood, orient="records", dtype={
                             "patient_id": str})
        blood_count = blood["patient_id"].value_counts().reset_index()
        blood_count.columns = ["patient_id", "Blood data"]
        self.assertEqual(np.sum(blood_reader_data["Blood data"]), np.sum(
            blood_count["Blood data"]), "The blood data is not the same.")
    
    def test(self):
        self.test_blood_data_frame_reader()
        print("Passed Blood data test.")
        print("All tests passed.")

if __name__ == '__main__':
    TestDataReader().test()