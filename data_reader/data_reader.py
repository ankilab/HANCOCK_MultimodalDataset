from pathlib import Path
import pandas as pd
import os
import re
import sys
import numpy as np

sys.path.append(str(Path(__file__).parents[1]))
from defaults import (
    DefaultPaths,
    DefaultFileNames
)


defaultPaths = DefaultPaths()
defaultFileNames = DefaultFileNames()


# noinspection PyDefaultArgument
class DataFrameReader:
    """The DataFrameReader class reads the HANCOCK data from the directory.
    All individual DataFrameReader's should inherit from this class and implement
    the return_data method.

    Methods:
        return_data: Returns the data as pandas DataFrame.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the data directory.
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
        returns the data in a pandas DataFrame. For image data it returns the
        patient_id and the path to the image.

        Args:
            data_dir (Path, optional): The absolute path of the file in the case 
            of tabular structured data or the starting directory in case 
            of the image data or textual report data. Defaults to Path(__file__).
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
        return self.data

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


class JsonDataFrameReader(DataFrameReader):
    """The DataFrameReader for the tabular structured data, clinical, 
    blood and pathological data. It reads a json file from the specified
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


class CSVDataFrameReader(JsonDataFrameReader):
    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_csv(self._data_dir, dtype={'patient_id': str})
        return self._data.copy()


# noinspection PyDefaultArgument
class FileRelationDataFrameReader(DataFrameReader):
    """The DataFrameReader for the data that is structured in several files and
    where we only want to create a relation between the patient_id and the file.
    Only the files in the initializer specified directory are considered.
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

    def _get_patient_id_to_file_relation_single_dir(
            self, data_dir_path: Path = None
    ) -> pd.DataFrame:
        """Creates a pandas data frame that contains the patient_id and the 
        file path in two columns. Only the files in the initializer
        specified directory are considered.

        Args:
            data_dir_path(Path | None, optional): The base directory where the
            files should be searched. If None, the directory given in the constructor
            will be used.

        Returns:
            pd.DataFrame: Data frame with two columns, patient_id and file.
        """
        use_data_dir = self._data_dir if data_dir_path is None else data_dir_path
        file_list = []
        for file_name in os.listdir(use_data_dir):
            if not os.path.isdir(use_data_dir / file_name):
                patient_id = re.search(r"[0-9]{3}", file_name).group()
                file_list.append({
                    self._columns[0]: patient_id,
                    self._columns[1]: use_data_dir / file_name
                })
        if not file_list:
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

    def return_data_count(self, columns: list[str] = ['patient_id', 'file']):
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
        return super().return_data_count(columns)


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


class NpzDataFrameReader(DataFrameReader):
    """The DataFrameReader for npz files.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data: The data from the npz file as pandas dataframe.
    """
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self._get_data()
        return self._data.copy()

    def __init__(self, data_dir: Path = Path(__file__)):
        super().__init__(data_dir=data_dir)

    def _get_data(self) -> pd.DataFrame:
        """Returns the data from the npz file.
        """
        npz_data = np.load(str(self._data_dir), allow_pickle=True)
        patient_ids = npz_data.keys()

        data_ids = []
        for patient_id in patient_ids:
            data_ids.append(npz_data[patient_id])

        marker = self._get_marker_from_data_dir()

        return pd.DataFrame({'patient_id': patient_ids, marker: data_ids})

    def _get_marker_from_data_dir(self) -> str:
        """Returns the marker from the data directory. The file name should be
        something like 'centertile_dtr_256dim_{marker}.npz'.
        If the marker is not found in the file name, 'data' is returned.
        """
        stem = self._data_dir.stem
        stem_split = stem.split('_')
        marker = 'data'
        if len(stem_split) > 1:
            marker = stem_split[len(stem_split) -1 ].split('.')[0]
        return marker


# ----- DataReader for the data set -----
# noinspection PyDefaultArgument
class PathologicalDataFrameReader(JsonDataFrameReader):
    """DataReader for the pathological structured data.
    """

    def __init__(self, data_dir: Path = defaultPaths.patho):
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


# noinspection PyDefaultArgument
class ClinicalDataFrameReader(JsonDataFrameReader):
    """DataReader for the clinical structured data.
    """

    def __init__(self, data_dir: Path = defaultPaths.clinical):
        super().__init__(data_dir)

    def return_data_count(
            self, columns: list[str] = ['patient_id', 'Clinical data']):
        return super().return_data_count(columns)


# noinspection PyDefaultArgument
class BloodDataFrameReader(JsonDataFrameReader):
    """DataReader for the blood structured data.
    """

    def __init__(self, data_dir: Path = defaultPaths.blood):
        super().__init__(data_dir)

    def return_data_count(self, columns: list[str] = ['patient_id', 'Blood data']):
        return super().return_data_count(columns)


# noinspection PyDefaultArgument
class WSIPrimaryTumorDataFrameReader(FileRelationDataFrameReader):
    """DataReader for the WSI primary tumor data.
    """

    def __init__(self, data_dir: Path = defaultPaths.wsi_tumor):
        super().__init__(data_dir)
        self._columns = ['patient_id', 'WSI Primary tumor']

    def return_data_count(
            self, columns: list[str] = ['patient_id', 'WSI Primary tumor']):
        """Returns a pandas data frame only with the count of rows we have 
        for the first column. The second given column will be used as the 
        column name for the count.

        Args:
            columns (list[str], optional): The names of the columns of the output 
            data frame. First one is the value that will be counted, second 
            one will be used as the column name for the count. 
            Defaults to ['patient_id', 'WSI Primary tumor'].

        Returns:
            pd.DataFrame: The data frame with the count of the first column.
        """
        return super().return_data_count(columns)


# noinspection PyDefaultArgument
class WSILymphNodeDataFrameReader(FileRelationDataFrameReader):
    """DataReader for the WSI lymph node data.
    """

    def __init__(self, data_dir: Path = defaultPaths.wsi_lymph_node):
        super().__init__(data_dir)

    def return_data_count(
            self, columns: list[str] = ['patient_id', 'WSI Lymph node']):
        """Returns a pandas data frame only with the count of rows we have 
        for the first column. The second given column will be used as the 
        column name for the count.

        Args:
            columns (list[str], optional): The names of the columns of the output 
            data frame. First one is the value that will be counted, second 
            one will be used as the column name for the count. 
            Defaults to ['patient_id', 'WSI Lymph node'].

        Returns:
            pd.DataFrame: The data frame with the count of the first column.
        """
        return super().return_data_count(columns)


# noinspection PyDefaultArgument
class TextDataReportsDataFrameReader(FileRelationDataFrameReader):
    """DataReader for the textual report data.
    """

    def __init__(self, data_dir: Path = defaultPaths.reports):
        super().__init__(data_dir)

    def return_data_count(
            self, columns: list[str] = ['patient_id', 'Surgery report']):
        """Returns a pandas data frame only with the count of rows we have 
        for the first column. The second given column will be used as the 
        column name for the count.

        Args:
            columns (list[str], optional): The names of the columns of the output 
            data frame. First one is the value that will be counted, second 
            one will be used as the column name for the count. 
            Defaults to ['patient_id', 'WSI Lymph node'].

        Returns:
            pd.DataFrame: The data frame with the count of the first column.
        """
        return super().return_data_count(columns)


# noinspection PyDefaultArgument
class TMACellDensityDataFrameReader(JsonDataFrameReader):
    """Data Reader for the cell density measurements of the TMA data.
    """
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = pd.read_csv(self._data_dir, dtype={"Case ID": str})
        return self._data.copy()

    def __init__(
        self,
        data_dir: Path = defaultPaths.cell_density,
        tma_name: str = 'TMA CD3'
    ):
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


# ----- DataReader for the generated features -----
class FeatureClinicalDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.feature_clinical):
        super().__init__(data_dir)


class FeaturePathologicalDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.feature_patho):
        super().__init__(data_dir)


class FeatureBloodDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.feature_blood):
        super().__init__(data_dir)


class FeatureICDCodesDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.feature_icd_codes):
        super().__init__(data_dir)


class FeatureTMACellDensityDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.feature_cell_density):
        super().__init__(data_dir)


class FeatureTabularMergedDataFrameReader(DataFrameReader):
    @property
    def data(self):
        if self._data is None:
            self._data = self._merge_data()
        return self._data.copy()

    def __init__(self, data_dir: Path = defaultPaths.features):
        """Data frame reader for the feature data from clinical, pathological,
        blood, icd codes and tma cell density. The data is merged on 'patient_id'.
        Assumes that the file names are the same as in the DefaultPaths class.


        Args:
            data_dir (Path, optional): The data directory where the
            features and targets can be found after they are generated with
            the create_multimodal_patient_vectors.py.
            Defaults to defaultPaths.features.
        """
        super().__init__(data_dir)
        self._target = None
        self._prepare_data_reader()

    def _prepare_data_reader(self):
        self.feature_clinical_reader = FeatureClinicalDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_clinical
        )
        self.feature_patho_reader = FeaturePathologicalDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_patho
        )
        self.feature_blood_reader = FeatureBloodDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_blood
        )
        self.feature_icd_codes_reader = FeatureICDCodesDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_icd_codes
        )
        self.feature_cell_density_reader = FeatureTMACellDensityDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_cell_density
        )

    def _merge_data(self) -> pd.DataFrame:
        """
        Merges the feature data from clinical, pathological, blood, icd codes and
        tma cell density to a single dataframe on patient_id
        """
        df = self.feature_clinical_reader.data
        df = df.merge(self.feature_patho_reader.data,
                      on='patient_id', how='outer')
        df = df.merge(self.feature_blood_reader.data,
                      on='patient_id', how='outer')
        df = df.merge(self.feature_icd_codes_reader.data,
                      on='patient_id', how='outer')
        df = df.merge(self.feature_cell_density_reader.data,
                      on='patient_id', how='outer')
        df.reset_index(drop=True, inplace=True)

        return df

    def return_data(self) -> pd.DataFrame:
        """Returns the merged feature data from clinical, pathological, blood,
        icd codes and tma cell density in a single dataframe merged on 'patient_id'.

        Returns:
            pd.DataFrame: The merged feature data.
        """
        return self.data


class FeatureTmaCd3DataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_cd3):
        super().__init__(data_dir=data_dir)


class FeatureTmaCd8DataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_cd8):
        super().__init__(data_dir=data_dir)


class FeatureTmaCd56DataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_cd56):
        super().__init__(data_dir=data_dir)


class FeatureTmaCd68DataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_cd68):
        super().__init__(data_dir=data_dir)


class FeatureTmaCd163DataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_cd163):
        super().__init__(data_dir=data_dir)


class FeatureTmaHeDataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_he):
        super().__init__(data_dir=data_dir)


class FeatureTmaMhc1DataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_cd3):
        super().__init__(data_dir=data_dir)


class FeatureTmaPdl1DataFrameReader(NpzDataFrameReader):
    """Reader for the TMA CD3 center tile data.

    Methods:
        return_data: Returns the data from the npz file.
        return_data_count: Returns the count of the data for each patient_id.

    Properties:
        data (pd.DataFrame): The data from the npz file as pandas dataframe.
    """
    def __init__(self, data_dir: Path = defaultPaths.feature_tma_pdl1):
        super().__init__(data_dir=data_dir)


class FeatureTmaMergedDataFrameReader(DataFrameReader):
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self._get_data()
        return self._data.copy()

    def __init__(self, data_dir: Path = defaultPaths.features):
        super().__init__(data_dir=data_dir)

    def _get_data(self, clean_data:bool = True) -> pd.DataFrame:
        """Reads the data from the npz files and merges them on 'patient_id'.

        Args:
            clean_data (bool, optional): If True, the data will be cleaned,
            meaning that NaN values will be replaced with zero vectors.

        Returns:
            pd.DataFrame: The merged data from the npz files.
        """
        data_reader_list = self._create_data_reader_list()
        data = data_reader_list[0].return_data()
        for data_reader in data_reader_list[1:]:
            data = pd.merge(
                data,
                data_reader.return_data(),
                on='patient_id', how='outer'
            )

        if clean_data:
            data = self._clean_data(data)

        return data

    def _create_data_reader_list(self) -> list[DataFrameReader]:
        """Creates a list of data readers for the TMA data.
        Starts in the self._data_dir but the file names are
        taken from the defaultFileName class.

        Returns:
            list[DataFrameReader]: List of data readers for the
            TMA feature data.
        """
        data_reader_list = [FeatureTmaCd3DataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_cd3
        ), FeatureTmaCd8DataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_cd8
        ), FeatureTmaCd56DataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_cd56
        ), FeatureTmaCd68DataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_cd68
        ), FeatureTmaCd163DataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_cd163
        ), FeatureTmaHeDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_he
        ), FeatureTmaMhc1DataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_mhc1
        ), FeatureTmaPdl1DataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_tma_pdl1
        )]
        return data_reader_list

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned_data = data.map(self._replace_nan_with_zeros)
        return cleaned_data

    @staticmethod
    def _replace_nan_with_zeros(x):
        if isinstance(x, float) and np.isnan(x):
            return np.zeros(shape=(512, ))
        return x


class FeatureTmaAndTabularMergedDataFrameReader(DataFrameReader):
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self._get_data()
        return self._data.copy()

    def __init__(self, data_dir: Path = defaultPaths.features):
        super().__init__(data_dir=data_dir)

    def _get_data(self):
        feature_tma_reader = FeatureTmaMergedDataFrameReader(data_dir=self._data_dir)
        feature_tabular_reader = FeatureTabularMergedDataFrameReader(data_dir=self._data_dir)

        data = feature_tma_reader.return_data()
        data = data.merge(
            feature_tabular_reader.return_data(),
            on='patient_id', how='outer'
        )
        return data


class FeatureTmaAndTabularWithoutPathologicalTmaCellDensityDataFrameReader(
    DataFrameReader
):
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self._get_data()
        return self._data.copy()

    def __init__(self, data_dir: Path = defaultPaths.features):
        super().__init__(data_dir=data_dir)

    def _get_data(self):
        feature_tma_reader = FeatureTmaMergedDataFrameReader(data_dir=self._data_dir)
        feature_clinical_reader = FeatureClinicalDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_clinical)
        feature_blood_reader = FeatureBloodDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_blood)
        feature_icd_codes_reader = FeatureICDCodesDataFrameReader(
            data_dir=self._data_dir / defaultFileNames.feature_icd_codes)
        data = feature_clinical_reader.return_data()
        data = data.merge(
            feature_blood_reader.return_data(),
            on='patient_id', how='outer')
        data = data.merge(
            feature_icd_codes_reader.return_data(),
            on='patient_id', how='outer')
        data = data.merge(
            feature_tma_reader.return_data(),
            on='patient_id', how='outer')
        return data


# --- DataReader for the targets ---
class TargetsDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.targets):
        super().__init__(data_dir)


class TargetsAdjuvantPredictionDataFrameReader(CSVDataFrameReader):
    @property
    def data(self):
        if self._data is None:
            self._data = self._create_data()
        return self._data.copy()

    def __init__(self, data_dir: Path = defaultPaths.targets):
        super().__init__(data_dir)

    def _create_data(self):
        data = super().data
        data.survival_status = (data.survival_status == 'deceased').astype(int)
        data.recurrence = (data.recurrence == 'yes').astype(int)
        data['followup_months'] = self._days_to_months(days=data['days_to_last_information'])
        data['months_to_rfs_event'] = self._days_to_months(days=data['days_to_rfs_event'])
        return data

    @staticmethod
    def _days_to_months(days: np.array) -> np.array:
        avg_days_per_month = 365.25 / 12
        return np.round(days / avg_days_per_month)


# -- DataReader for data splits
class DataSplitBloodDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.data_split_blood):
        super().__init__(data_dir)


class DataSplitClinicalDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.data_split_clinical):
        super().__init__(data_dir)


class DataSplitPathologicalDataFrameReader(CSVDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.data_split_patho):
        super().__init__(data_dir)


class DataSplitInDataFrameReader(JsonDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.data_split_in):
        super().__init__(data_dir)


class DataSplitOropharynxDataFrameReader(JsonDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.data_split_oropharynx):
        super().__init__(data_dir)


class DataSplitOutDataFrameReader(JsonDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.data_split_out):
        super().__init__(data_dir)


class DataSplitTreatmentOutcomeDataFrameReader(JsonDataFrameReader):
    def __init__(self, data_dir: Path = defaultPaths.data_split_treatment_outcome):
        super().__init__(data_dir)
