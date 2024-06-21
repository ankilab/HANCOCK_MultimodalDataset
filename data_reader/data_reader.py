from pathlib import Path
import pandas as pd
import os
import re


class DataFrameReader:
    """The DataFrameReader class reads the HANCOCK data from the directory.
    All individual DataFrameReader's should inherit from this class and implement
    the return_data method.
    """
    @property
    def data(self) -> pd.DataFrame:
        """The getter for the data property. The data property should be 
        implemented by the inheriting classes.
        """
        if self._data is None:
            self._data = pd.DataFrame(columns=['patient_id', 'data'])
        return self._data.copy()

    def __init__(self, data_dir: Path = Path(__file__)):
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
        self._data = None

    def return_data(self) -> pd.DataFrame:
        """Returns the data from the data directory.

        Returns:
            pd.DataFrame: The data in a pandas dataframe.

        Throws:
            FileNotFoundError: If the file is not found at the in the initializer
            specified path, the FileNotFoundError is raised.
        """
        return pd.DataFrame(columns=['patient_id', 'data'])

    def return_data_count(self, columns: list[str] = ['patient_id', 'data']):
        """Returns a pandas data frame only with the count of rows we have 
        for the first column. The second given column will be used as the 
        column name for the count.

        Args:
            columns (list[str], optional): The names of the columns of the output 
            data frame. First one is the value that will be counted, second 
            one will be used as the column name for the count. 
            Defaults to ['patient_id', 'data'].

        Returns:
            pd.DataFrame: The data frame with the count of the first column.
        """
        data = self.data
        data_count = data[columns[0]].value_counts().reset_index()
        data_count.columns = columns
        return data_count


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
        return self._data.copy()

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)

    def return_data(self) -> pd.DataFrame:
        """Returns the data as pandas data frame that is located at the 
        path given with the data_dir parameter in the initializer.

        Returns:
            pd.DataFrame: _description_
        """
        return self.data


class FileRelationDataFrameReader(DataFrameReader):
    """The DataFrameReader for the data that is structured in several files and
    where we only want to create a relation between the patient_id and the file.
    Only the files in the in the initializer specified directory are considered.
    """
    @property
    def data(self) -> pd.DataFrame:
        """The getter for the data frame that contains the patient_id to 
        file relation.
        """
        if self._data is None:
            self._data = self._get_patient_id_to_file_relation_single_dir()
        return self._data.copy()

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir=data_dir)
        self._columns = ['patient_id', 'file']

    def _get_patient_id_to_file_relation_single_dir(self) -> pd.DataFrame:
        """Creates a pandas data frame that contains the patient_id and the 
        file path in two columns. Only the files in the in the initializer 
        specified directory are considered.

        Returns:
            pd.DataFrame: Data frame with two columns, patient_id and file.
        """
        file_list = []
        for file_name in os.listdir(self._data_dir):
            if not os.path.isdir(self.data_dir / file_name):
                patient_id = re.search(r"[0-9]{3}", file_name).group()
                file_list.append({
                    self._columns[0]: patient_id,
                    self._column[1]: self._data_dir / file_name
                })
        if (file_list == []):
            slide_df = pd.DataFrame(columns=self._columns)
        else:
            slide_df = pd.DataFrame(file_list)
        return slide_df

    def return_data(self) -> pd.DataFrame:
        """Returns a pandas data frame that contains the patient_id and the
        file in relation to each other. Only a copy of the original data is
        returned. If you want to update the data, because some files were added
        you have to reinitialize the object.

        Returns:
            pd.DataFrame: The data frame with the patient_id and the file.
        """
        return self.data


class SubDirDataFrameReader(FileRelationDataFrameReader):
    """The DataFrameReader for the data that is structured in several files and
    where we only want to create a relation between the patient_id and the file.
    Also files that are located in subdirectories of the in the initializer 
    specified directory are considered.
    """
    @property
    def data(self) -> pd.DataFrame:
        """The getter for the data frame that contains the patient_id to 
        file relation.
        """
        if self._data is None:
            self._data = self._get_patient_id_to_file_relation_sub_dir()
        return self._data.copy()

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir=data_dir)

    def _get_patient_id_to_file_relation_sub_dir(self) -> pd.DataFrame:
        """Creates a pandas data frame that contains the patient_id and the 
        file path in two columns. 
        Also files that are located in subdirectories of the in the initializer 
        specified directory are considered.

        Returns:
            pd.DataFrame: Data frame with two columns, patient_id and file.
        """
        slide_df = pd.DataFrame(columns=self._columns)
        for sub_dir in os.listdir(self._data_dir):
            dir_data_frame = self._get_patient_id_to_file_relation_single_dir(
                self._data_dir / sub_dir)
            slide_df = pd.concat([slide_df, dir_data_frame], ignore_index=True)
        return slide_df

    def return_data(self) -> pd.DataFrame:
        """Returns a pandas data frame that contains the patient_id and the
        file in relation to each other. Only a copy of the original data is
        returned. If you want to update the data, because some files were added
        you have to reinitialize the object.

        Returns:
            pd.DataFrame: The data frame with the patient_id and the file.
        """
        return self.data


class PathologicalDataFrameReader(TabularDataFrameReader):
    """DataReader for the pathological structured data.
    """

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)

    def return_data_count(
        self, columns: list[str] = ['patient_id', 'Pathological data']
    ) -> pd.DataFrame:
        """Returns a pandas data frame only with the count of rows we have 
        for the first column. The second given column will be used as the 
        column name for the count.

        Args:
            columns (list[str], optional): The names of the columns of the output 
            data frame. First one is the value that will be counted, second 
            one will be used as the column name for the count. 
            Defaults to ['patient_id', 'Pathological data'].

        Returns:
            pd.DataFrame: The data frame with the count of the first column.
        """
        return super().return_data_count(columns)


class ClinicalDataFrameReader(TabularDataFrameReader):
    """DataReader for the clinical structured data.
    """

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)

    def return_data_count(self, columns: list[str] = ['patient_id', 'Clinical data']):
        return super().return_data_count(columns)


class BloodDataFrameReader(TabularDataFrameReader):
    """DataReader for the blood structured data.
    """

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)

    def return_data_count(self, columns: list[str] = ['patient_id', 'Blood data']):
        return super().return_data_count(columns)


class WSIPrimaryTumorDataFrameReader(SubDirDataFrameReader):
    """DataReader for the WSI primary tumor data.
    """

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)


class WSILymphNodeDataFrameReader(FileRelationDataFrameReader):
    """DataReader for the WSI lymph node data.
    """

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir)


class TMACellDensityDataFrameReader(TabularDataFrameReader):
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            # As the Case Id is not unique in the data I assume that
            # they are equivalent with the patient_id.
            self._data = pd.read_csv(self._data_dir, type={"Case ID": str})
        return self._data.copy()

    def __init__(self, data_dir: Path = Path(__file__), tma_name: str = 'TMA CD3'):
        super().__init__(data_dir)
        self._tma_name = tma_name

    def return_data_count(self, columns: list[str] = ['Case ID', 'Image']):
        """Returns the count of the Case ID in the data frame (or the first column if 
        you change that) and analyses the 'Image' column (or the second column if 
        you change that) if it includes the location (either TumorCenter InvasionCenter) 
        in the value of each row. The count of both is returned in a pandas data frame
        with the columns 'patient_id', self._tma_name + ' tumor center' 
        and self._tma_name + ' invasion center'.

        Args:
            columns (list[str], optional): _description_. Defaults to ['Case ID', 
            'location', 'Image'].
        """
        data = self.data
        data = data[~(data['Missing'] == True) & ~data[columns[0]].isna()]
        data['location'] = data[columns[1]].str.extract(
            r"(TumorCenter|InvasionFront)"
        )
        tma_z = data[data['location'] == 'TumorCenter']
        tma_inv = data[data['location'] == 'InvasionFront']
        count_z = tma_z[columns[0]].value_counts().reset_index()
        count_inv = tma_inv[columns[0]].value_counts().reset_index()
        count_z.columns = ["patient_id", self._tma_name + " tumor center"]
        count_inv.columns = ["patient_id", self._tma_name + " invasion front"]
        complete_count = pd.merge(
            count_z, count_inv, on="patient_id", how="outer")
        
        return complete_count
