import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from multimodal_machine_learning.predictor.adjuvant_treatment import (
    adjuvant_treatment_prediction_tma_vector_attention
)


if __name__ == '__main__':
    adjuvant_treatment_prediction_tma_vector_attention(save_flag=True, plot_flag=True)
