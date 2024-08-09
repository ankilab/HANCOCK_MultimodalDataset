import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from multimodal_machine_learning.predictors import (
    TmaTabularMergedAttentionMlpAdjuvantTreatmentPredictor,
    TmaAttentionMlpAdjuvantTreatmentPredictor,
    TmaTabularMergedWithoutPathologicalDataTmaCellDensityAdjuvantTreatmentPredictor
)


def adjuvant_treatment_prediction_tma_vector_attention(
        save_flag=True, plot_flag=True
):
    predictors = [
        TmaTabularMergedAttentionMlpAdjuvantTreatmentPredictor(
            save_flag=save_flag, plot_flag=plot_flag
        ),
        TmaAttentionMlpAdjuvantTreatmentPredictor(
            save_flag=save_flag, plot_flag=plot_flag
        ),
        TmaTabularMergedWithoutPathologicalDataTmaCellDensityAdjuvantTreatmentPredictor(
            save_flag=save_flag, plot_flag=plot_flag
        )
    ]
    plot_base_name = 'attention_mlp_'
    predictor_names = [
        'tma_tabular_merged',
        'tma',
        'tma_tabular_merged_without_patho_tma_cell_density'
    ]
    plot_lime = [
        True,
        False,
        False
    ]

    for i, (predictor, predictor_name) in enumerate(zip(predictors, predictor_names)):
        print(f'Running training for {predictor_name} data ...')
        _ = predictor.train(plot_name=plot_base_name + predictor_name + '_train', lime_flag=plot_lime[i])
        print(f'Running k-fold cross-validation for {predictor_name} data ...')
        _ = predictor.cross_validate(plot_name=plot_base_name + predictor_name + '_cross_val')


if __name__ == '__main__':
    adjuvant_treatment_prediction_tma_vector_attention(save_flag=True, plot_flag=True)
