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

### Train models
```bash
python src/train.py --config experiments/config.yaml
```

### Evaluate metrics (MAE, sMAPE, DTW for Figure 5)
```bash
python src/evaluate.py --config experiments/config.yaml
```

### Measure training and inference time (Figure 4)
```bash
python src/evaluate.py --time-benchmark
```

### Detect stability points and delay (Figure 7)
```bash
python src/detect_stability.py
```

### Analyze forecast variance (Figure 8)
```bash
python src/analyze_variance.py
```

### Reproduce all figures from the paper
```bash
python src/plot_figures.py
```

## 📊 Summary of Reproducible Results

| **Result**                      | **Script**                  | **Figure**  |
|---------------------------------|-----------------------------|-------------|
| Model performance metrics       | `evaluate.py`               | Figure 5    |
| Inference and training time     | `evaluate.py --time-benchmark` | Figure 4    |
| Stability detection alignment   | `detect_stability.py`       | Figure 7    |
| Forecast variance and std dev   | `analyze_variance.py`       | Figure 8    |
| Forecast visualizations         | `plot_figures.py`           | Figures 3, 6, 10, 11 |
| Domain-specific anomaly (E1)    | `plot_figures.py`           | Figure 12   |

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.
