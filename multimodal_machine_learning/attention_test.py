import numpy as np
import tensorflow as tf
from data_reader import DataFrameReaderFactory
from multimodal_machine_learning.predictors import TmaTabularMergedAttentionMlpAdjuvantTreatmentPredictor
from defaults import DefaultPaths


if __name__ == '__main__':
    input_dim = 8 * 512 + 104
    factory = DataFrameReaderFactory()

    predictor = TmaTabularMergedAttentionMlpAdjuvantTreatmentPredictor(plot_flag=True, save_flag=False)
    train_return = predictor.train(plot_name='attention_mlp_tma_tabular_merged_train', lime_flag=False, epochs=10)

    attention_layer = predictor.model.layers[1]

    attention_scores = []
    for sample in np.concatenate([train_return[1][2], train_return[1][0]], axis=0):
        attention_scores.append(attention_layer(
            tf.expand_dims(sample, axis=0), training=False,
            return_attention_scores=False)[1])

    attention_scores = [np.array(lst) for lst in attention_scores]
    data = {f'array_{i}': arr for i, arr in enumerate(attention_scores)}
    npz_path = str(DefaultPaths().results / 'attention_scores.npz')
    np.savez(npz_path, **data)
    loaded_data = np.load(npz_path)

    # Extract arrays and stack them
    arrays = [loaded_data[key] for key in sorted(loaded_data.keys())]
    combined_array = np.stack(arrays, axis=0)  # Shape will be (number_of_arrays, array_shape)

    print("Combined array shape:", combined_array.shape)
