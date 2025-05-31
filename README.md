# Reproducibility Package for "Forecasting Stability in Tritium Source Monitoring"

This repository provides code and configuration to reproduce the main results from our ICDM 2025 paper. It includes model training, performance evaluation, stability detection, and visualization components.

## 📁 Project Structure

```
.
├── src/
│   ├── models/              # Model definitions: LSTM, N-BEATS, TFT, etc.
│   ├── utils/               # Utility functions (filtering, metrics, plotting)
│   ├── train.py             # Train models using config
│   ├── evaluate.py          # Evaluate models, compute MAE/sMAPE/DTW
│   ├── detect_stability.py  # Detect stability point Xc from time series
│   ├── analyze_variance.py  # Analyze prediction variance across runs
│   └── plot_figures.py      # Generate plots for all paper figures
├── experiments/
│   ├── config.yaml          # Model hyperparameters and run settings
│   └── run_all.sh           # Shell script to execute all experiments
├── data/                    # Folder for input datasets or data download link
├── outputs/                 # Folder to store figures, logs, and model outputs
└── requirements.txt         # Python package dependencies
```

## 🔧 Setup Instructions

### 1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

Create the models folder first if you haven't.

### Train models
```bash
python src/train.py --config experiments/lstm_config.yaml
python src/train.py --config experiments/nlinear_config.yaml
python src/train.py --config experiments/dlinear_config.yaml
python src/train.py --config experiments/nhits_config.yaml
python src/train.py --config experiments/nbeats_config.yaml
python src/train.py --config experiments/tft_config.yaml
python src/train.py --config experiments/tsmixer_config.yaml
```
If you cannot train the models, you can download the pre-trained models here (3.18 GB): https://drive.google.com/file/d/1I3CY1sfkoOdHgKDYbJLbW8IENqc3kwUh/view?usp=sharing

### Evaluate metrics (MAE, sMAPE, DTW for Figure 4)
```bash
python src/evaluate_mae_smape_dtw_pre.py
```
The `pre` suffix in the script indicates the pre-run values stored as a dictionary. If you're more interested in generating your own values, you can use the similar script with the `fresh` suffix.

It will output three images: mae.png, smape.png, and dtw.png.

### Measure training and inference time (Figure 9a and 9b)
```bash
python src/evaluate_time.py
```
This was done manually, in the code to generate `eval_list`, we use 

```python
start_time = time.time()  # Start time
...
...
end_time = time.time()    # End time
print(f"Time taken: {end_time - start_time:.6f} seconds")
```

### Detect stability points and delay (Figure 6)
```bash
python src/detect_stability.py
```

### Analyze forecast variance (Figure 7)
```bash
python src/analyze_rmse.py
python src/analyze_std_dev.py
python src/analyze_variance_moving_prediction.py
```

### Reproduce all figures from the paper
```bash
python src/plot_figures.py
```

## 📊 Summary of Reproducible Results

| **Result**                      | **Script**                  | **Figure**  |
|---------------------------------|-----------------------------|-------------|
| Model performance metrics       | `evaluate_mae_smape_dtw_pre.py`               | Figure 4    |
| Inference and training time     | `evaluate_time.py` | Figures 9a and 9b    |
| Stability detection alignment   | `detect_stability.py`       | Figure 6    |
| Forecast RMSE   | `analyze_rmse.py`       | Figure 7a    |
| Forecast std dev   | `analyze_std_dev.py`       | Figure 7b    |
| Forecast variance   | `analyze_variance_moving_prediction.py`       | Figure 7c    |
| Forecast visualizations         | `plot_figures.py`           | Figures 3, 5, 9c, 10 |
| Domain-specific anomaly (E1)    | `plot_figures.py`           | Figure 11   |

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
