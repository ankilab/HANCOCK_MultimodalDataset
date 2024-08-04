from multimodal_machine_learning.predictors import (
    TabularMergedAdjuvantTreatmentPredictor, ClinicalAdjuvantTreatmentPredictor,
    PathologicalAdjuvantTreatmentPredictor, BloodAdjuvantTreatmentPredictor,
    TMACellDensityAdjuvantTreatmentPredictor, ICDCodesAdjuvantTreatmentPredictor
)


def adjuvant_treatment_prediction_tabular_only(save_flag=True, plot_flag=True):
    multi_predictor = TabularMergedAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag)
    clinical_predictor = ClinicalAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag)
    patho_predictor = PathologicalAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag)
    blood_predictor = BloodAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag)
    tma_cell_density_predictor = TMACellDensityAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag)
    icd_predictor = ICDCodesAdjuvantTreatmentPredictor(
        save_flag=save_flag, plot_flag=plot_flag)
    cross_validation_splits = 10

    data_types = [
        'multimodal', 'clinical', 'pathological', 'blood', 'cell density', 'text'
    ]
    predictors = [
        multi_predictor, clinical_predictor, patho_predictor, blood_predictor,
        tma_cell_density_predictor, icd_predictor
    ]

    for data_type, predictor in zip(data_types, predictors):
        print(f'Running k-fold cross-validation for {data_type} data ...')
        _ = predictor.cross_validate(n_splits=cross_validation_splits,
                                     plot_name='adjuvant_therapy_' + data_type)

    # Train classifier once on multimodal data, show survival curves and bar plot
    print("Training and testing the final multimodal model...")
    multi_predictor.train(
        plot_name='adjuvant_therapy_multimodal', df_train=multi_predictor.df_train,
        df_other=multi_predictor.df_test
    )


if __name__ == '__main__':
    adjuvant_treatment_prediction_tabular_only(save_flag=True, plot_flag=True)
