from pathlib import Path
import pandas as pd


class DataFrameReader:
    """The DataFrameReader class reads the HANCOCK data from the directory.
    All individual DataFrameReader's should inherit from this class and implement
    the return_data method.
    """

    def __init__(self, data_dir: Path = Path(__file__), include_sub_dir: bool = False):
        """The DataReader class reads the HANCOCK data from the directory. It
        return the data in a pandas DataFrame. For image data it returns the
        patient_id and the path to the image.

        Args:
            data_dir (Path, optional): The absolute path of the file in the case 
            of tabular structured data or the starting directory in case 
            of the image data or textual report data. Defaults to Path(__file__).
            include_sub_dir (bool, optional): For the image data, if the images are
            in subdirectories, set this to True to find all the images. Defaults
            to False.
        """
        self._data_dir = data_dir
        self._include_sub_dir = include_sub_dir

    def return_data(self) -> pd.DataFrame:
        """Returns the data from the data directory.

        Returns:
            pd.DataFrame: The data in a pandas dataframe.

        Throws:
            FileNotFoundError: If the file is not found at the in the initializer
            specified path, the FileNotFoundError is raised.
        """
        return pd.DataFrame(columns=['patient_id', 'data'])


class TabularDataFrameReader(DataFrameReader):
    """The DataFrameReader for the tabular structured data, clinical, 
    blood and pathological data. It reads an json file from the specified
    data directory and returns it as pandas dataframe.
    """
    @property
    def data(self):
        """The getter for the tabular_structured data."""
        if self._data is None:
            self._data = pd.read_json(
                self._data_dir, orient="records", dtype={"patient_id": str})

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self._data = None

    def return_data(self) -> pd.DataFrame:
        return self.data


class PathologicalDataFrameReader(TabularDataFrameReader):
    """DataReader for the pathological structured data.
    """
    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)


class ClinicalDataFrameReader(TabularDataFrameReader):
    """DataReader for the clinical structured data.
    """
    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)
        

class BloodDataFrameReader(TabularDataFrameReader):
    """DataReader for the blood structured data.
    """
    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)
