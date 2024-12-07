import os
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
from argument_parser import HancockArgumentParser
from data_reader import DataFrameReaderFactory


def count_words(file_path: Path):
    """
    Returns the number of words in a text file.

    file_path (pathlib.Path): Path of the text file to be read.
    """
    with open(file_path, 'r') as file:
        text = file.read()
        return len(text.split())
    
    
def assign_stage(row: pd.Series) -> int | None:
    """
    Assigns an integer value to the pT stage of the tumor. There are 4 stages
    that are encoded. The row should contain a column 'pT_stage'.

    Args:
        row (pd.Series): A row of a pandas DataFrame. The row should 
        contain a column 'pT_stage'.
    """
    stage_1 = ['pT1', 'pT1a', 'pT1b']
    stage_2 = ['pT2']
    stage_3 = ['pT3']
    stage_4 = ['pT4a', 'pT4b']
    if row['pT_stage'] in stage_1:
        return 1
    elif row['pT_stage'] in stage_2:
        return 2
    elif row['pT_stage'] in stage_3:
        return 3
    elif row['pT_stage'] in stage_4:
        return 4
    else:
        return None


class CorrelationSurgeryReportWordCountAndPtStage():
    """
    This class calculates the correlation coefficient between the word count 
    of the surgery report in german and the tumor pT stage.
    """
    def __init__(self):
        self.argumentParser = HancockArgumentParser(file_type='grading_correlation')
        self.args = self.argumentParser.parse_args()
        self._data_reader_factory = DataFrameReaderFactory()
        self.data_reader_text = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.text_reports,
            data_dir=self.args.path_reports,
            data_dir_flag=True
        )
        self.data_reader_patho = self._data_reader_factory.make_data_frame_reader(
            data_type=self._data_reader_factory.data_reader_types.patho,
            data_dir=self.args.path_patho,
            data_dir_flag=True
        )
        self._correlation_coefficient = None
        self._data_frame = None
        
    def _create_data_frame(self) -> pd.DataFrame:
        """
        Helper function to create the data frame that contains the pT 
        stage encoded in a single integer column 'stage' and the word count.
        """
        df_text = self.data_reader_text.return_data()
        df_patho = self.data_reader_patho.return_data()
        df_new = df_text.copy()
        df_new['word_count'] = df_new['file'].apply(count_words)
        df_new = pd.merge(df_new, df_patho[['patient_id', 'pT_stage']], 
                            on='patient_id', how='left')
        df_new['stage'] = df_new.apply(assign_stage, axis=1)
        df_new.drop(['file', 'pT_stage'], axis=1, inplace=True)
        df_new = df_new.dropna(subset=['stage'])        
        return df_new
    
    def return_correlation_score(self, method: str = 'spearman') -> float:
        """
        Returns the correlation coefficient between the word count of the
        surgery report in german and the tumor pT stage with the specified
        correlation method. Valid options are {‘pearson’, ‘kendall’, ‘spearman’}

        Args:
            method (str, optional): The correlation method used. 
            Defaults to 'spearman'.
        """
        if self._data_frame is None:
            self._data_frame = self._create_data_frame()
            
        self._correlation_coefficient = self._data_frame['stage'].corr(
            self._data_frame['word_count'], method=method
        )
        return self._correlation_coefficient
        

if __name__ == '__main__':
    correlation = CorrelationSurgeryReportWordCountAndPtStage()
    pearson = correlation.return_correlation_score(method='pearson')
    spearman = correlation.return_correlation_score(method='spearman')
    correlation_data = pd.DataFrame({'Coefficient': ['Spearman', 'Pearson'],
                                     'Value': [round(spearman, 4), 
                                               round(pearson, 4)]})
    print(correlation_data.to_markdown(index=False))
