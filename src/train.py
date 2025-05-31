
import yaml
import argparse
import numpy as np
from darts.models import RNNModel, NLinearModel
from darts.dataprocessing.transformers import Scaler
from preprocessing import process_zip_and_generate_series

def load_series(zip_path, apply_smoothing=False, window_length=250, poly_order=3):
    return process_zip_and_generate_series(
        zip_path=zip_path,
        apply_smoothing=apply_smoothing,
        window_length=window_length,
        poly_order=poly_order
    )

def build_model(cfg):
    m = cfg['model']
    model_type = m['type'].lower()

    if model_type == "lstm":
        return RNNModel(
            model="LSTM",
            input_chunk_length=m['input_chunk_length'],
            training_length=m['training_length'],
            n_epochs=m['n_epochs'],
            batch_size=m['batch_size'],
            dropout=m['dropout'],
            random_state=m['random_state']
        )
    elif model_type == "nlinear":
        return NLinearModel(
            input_chunk_length=m['input_chunk_length'],
            output_chunk_length=m['output_chunk_length'],
            n_epochs=m['n_epochs'],
            batch_size=m['batch_size'],
            random_state=m['random_state']
        )
    else:
        raise ValueError(f"Unsupported model type: {m['type']}")

def main(config_path):
    import os
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    smoothing_options = [None, 25, 50, 100, 250]
    for win_len in smoothing_options:
        print(f"Training with smoothing: {win_len}")
        series = load_series(
            zip_path=cfg['data']['zip_path'],
            apply_smoothing=win_len is not None,
            window_length=win_len if win_len is not None else 0,
            poly_order=cfg['data'].get('smooth_poly', 3)
        )

        series = [ts.astype(np.float32) for ts in series] if cfg['data'].get('force_float32', True) else series

        if cfg['data'].get('scale', True):
            scaler = Scaler()
            scaled_series_list = [scaler.fit_transform(s) for s in series]
        else:
            scaled_series_list = series

        indices = cfg['training']['train_series_index']
        train_series = [scaled_series_list[i] for i in indices]

        model = build_model(cfg)
        model.fit(train_series)

        model_tag = "nosmooth" if win_len is None else str(win_len)
        out_path = f"./models/{cfg['experiment_name']}_{model_tag}_model.pth"
        model.save(out_path)
        print(f"Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
