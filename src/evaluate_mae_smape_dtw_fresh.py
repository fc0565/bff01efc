import os
import shutil
import pwlf
import zipfile
#import patoolib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts.metrics import smape, mae, dtw_metric
# we don't want to penalize errors that hard, focussed more on the temporal
# trend

from darts.models import RNNModel, NLinearModel, DLinearModel, NHiTSModel, NBEATSModel, TFTModel, TSMixerModel

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from pathlib import Path

import numpy as np
import statistics
from collections import defaultdict

def get_ref_stable_point():
    ref_equi = {
        "25": "2023-11-13 20:05:58",
        "26": "2023-12-07 23:51:31",
        "27": "2024-02-06 18:25:34",
        "28": "2024-02-26 15:18:22",
        "29": "2024-05-13 16:46:30",
        "30": "2024-06-10 18:25:15",
        "31": "2024-06-11 18:30:10",
        "32": "2024-06-12 13:23:20"
    }
    return ref_equi

def load_model(name):
    model_path = "models"

    model_mapping = {
        "lstm": ("lstm", RNNModel),
        # ... other models omitted for brevity ...
        "tsmixer_250": ("tsmixer_250", TSMixerModel)
    }

    name = name.lower()
    if name not in model_mapping:
        print("Model not found")
        return None

    subfolder, model_class = model_mapping[name]
    model_path_full = f"{model_path}/{subfolder}_model.pth"

    model = model_class.load(model_path_full)
    if hasattr(model, "model"):
        model.model = model.model.float()
    return model

def process_zips(zip_path):
    dataframes = []
    with zipfile.ZipFile(zip_path, 'r') as zfile:
        csv_files = [f for f in zfile.namelist() if f.endswith('.csv')]
        for csv_file in csv_files:
            with zfile.open(csv_file) as file:
                df = pd.read_csv(file, index_col='timestamp')
                df.index = pd.to_datetime(df.index)
                dataframes.append(df)
    return dataframes

def find_stable_point(series, model):
    smoothed_values = series.values.astype(np.float32)
    numeric_index = (series.index.astype('int64') // 10**9).astype(np.float32)

    model = pwlf.PiecewiseLinFit(numeric_index, smoothed_values)
    breaks = model.fit(2)

    values = [model.predict([item])[0] for item in breaks]
    nick_point = breaks[0] if abs(float(values[1] - values[0])) < 2 else breaks[1]
    return nick_point

def regularize_data(data):
    most_common_interval = 100
    most_common_timedelta = pd.to_timedelta(most_common_interval, unit='s')
    regularized_data = data.resample(most_common_timedelta).bfill()
    return regularized_data

def eval_model(model, store_images=False, model_name="default", store_csv=False):
    predict_length = 460
    res = {"equi": [], "smape": [], "mae": [], "dtw": []}
    result_path = f"result"

    for i in range(25, len(scaled_series_list)):
        if i == 32:
            predict_length = 365

        first_36_values = scaled_series_list[i][:36].astype(np.float32)
        complete_original = scaler.inverse_transform(scaled_series_list[i]).values().flatten().astype(np.float32)
        first_36_values_original = scaler.inverse_transform(first_36_values).values().flatten().astype(np.float32)

        scaled_prediction = model.predict(n=predict_length, series=first_36_values).astype(np.float32)
        forecast_rnn_lstm = scaler.inverse_transform(scaled_prediction).astype(np.float32)

        original_timestamps = series[i].time_index
        original_values_series = pd.Series(complete_original, index=original_timestamps)
        input_values_series = pd.Series(first_36_values_original, index=original_timestamps[:36])
        predicted_values_series = pd.Series(forecast_rnn_lstm.values().flatten(), index=original_timestamps[36:36 + predict_length])

        combined_series = pd.concat([input_values_series, predicted_values_series])

        smape_value = smape(scaled_series_list[i][36:36+predict_length], scaled_prediction)
        mae_value = mae(scaled_series_list[i][36:36+predict_length], scaled_prediction)
        dtw_value = dtw_metric(scaled_series_list[i][36:36+predict_length], scaled_prediction)
        equi = find_stable_point(combined_series, model)

        res["equi"].append(equi)
        res["smape"].append(smape_value)
        res["mae"].append(mae_value)
        res["dtw"].append(dtw_value)

        if store_images:
            Path(result_path).mkdir(parents=True, exist_ok=True)
            original_values_series[:36 + predict_length].plot(label='actual')
            predicted_values_series.plot(label=model_name, lw=3)
            plt.legend()
            filename = f'result/{model_name}_predictions_{i}_{str(smape_value).replace(".", "_")}.png'
            plt.savefig(filename, format='png', dpi=300)
            plt.close()

        if store_csv:
            Path(result_path).mkdir(parents=True, exist_ok=True)
            filename = f'result/{model_name}_predictions_{i}.csv'
            combined_series.to_csv(filename)

    return res


# 1. slopes.zip
# 2. models.zip

zip_path = './data/slopes.zip'
dfs = process_zips(zip_path)

# unzip result.zip

# Combine the dataframes with their first timestamp index
dfs_with_timestamp = [(df, df.index[0]) for df in dfs]

# Sort the dataframes by the first timestamp in ascending order (oldest to most recent)
dfs_sorted = sorted(dfs_with_timestamp, key=lambda x: x[1])

# Extract sorted dataframes (ignoring the timestamps, which were used for sorting)
dfs1 = [df_tuple[0] for df_tuple in dfs_sorted]

# Regularize the dataframe timestamps (interval = 100s)
for i in range(len(dfs1)):
  dfs1[i] = regularize_data(dfs1[i])

# Remove datasets that are too small and egun-pusing event
index_to_remove = []

for i in range(len(dfs1)):
    #print(f"Length of DataFrame {i}: {len(dfs1[i])}")
    if len(dfs1[i]) < 496:
        index_to_remove.append(i)

#print(index_to_remove)

dataframes = [item for idx, item in enumerate(dfs1) if idx not in index_to_remove]

del dataframes[-1] # deleted last dataframe as it looks like an egun pulsing event (not what we're trying to predict here)

# Reprocessing the start of each event to better show the activity drift (heuristic)
dataframes[26] = dataframes[26].loc['2023-12-07 16:47:51':]
dataframes[27] = dataframes[27].loc['2024-02-06 12:54:58':]
dataframes[29] = dataframes[29].loc['2024-05-13 10:58:40':]
dataframes[31] = dataframes[31].loc['2024-06-11 14:30:43':]
dataframes[32] = dataframes[32].loc['2024-06-12 13:21:43':]

### Preparing Series for the models
# Convert to Darts TimeSeries object
series = []
for i in range(len(dataframes)):
    serie = TimeSeries.from_dataframe(dataframes[i],value_cols='value')
    series.append(serie)

# Scale each series in all_series using the same scaler
scaler = Scaler()
scaled_series_list = []

for serie in series:
    scaled_series = scaler.fit_transform(serie)
    scaled_series_list.append(scaled_series)

import time

"""
models = [
    "lstm", "lstm_25", "lstm_50", "lstm_100", "lstm_250",
    "nlinear", "nlinear_25", "nlinear_50", "nlinear_100", "nlinear_250",
    "dlinear", "dlinear_25", "dlinear_50", "dlinear_100", "dlinear_250",
    "nhits", "nhits_25", "nhits_50", "nhits_100", "nhits_250",
    "nbeats", "nbeats_25", "nbeats_50", "nbeats_100", "nbeats_250",
    "tft", "tft_25", "tft_50", "tft_100", "tft_250",
    "tsmixer", "tsmixer_25", "tsmixer_50", "tsmixer_100", "tsmixer_250"
]
"""

#models = ["tft", "tft2"]
models = ["lstm", "nlinear", "dlinear", "nhits", "nbeats", "tft", "tsmixer"]
#models = ["tsmixer"]

eval_list = []
start_time = time.time()  # Start time
for i in range(50):
    eval_results = {}
    eval_moving_results = {}
    for model_name in models:
        print(model_name)
        model = load_model(name=model_name)
        eval_results[model_name] = eval_model(model=model, store_images=False, model_name=model_name, store_csv=False)
    eval_list.append(eval_results)
end_time = time.time()    # End time
print(f"Time taken: {end_time - start_time:.6f} seconds")


# Define method to extract data for a given model
def summarize_eval_list(eval_list, model_name):
    # Collect metrics
    equi_list, mae_list, smape_list, dtw_list = [], [], [], []

    for item in eval_list:
        model_data = item[model_name]
        equi_list.append(model_data["equi"])
        mae_list.append(model_data["mae"])
        smape_list.append(model_data["smape"])
        dtw_list.append(model_data["dtw"])

    # Transpose and average
    equi_avg = [statistics.mean(e) for e in zip(*equi_list)]
    equi_stdev = [statistics.stdev(e) for e in zip(*equi_list)]
    mae_avg = [statistics.mean(m) for m in zip(*mae_list)]
    smape_avg = [statistics.mean(s) for s in zip(*smape_list)]
    dtw_avg = [statistics.mean(d) for d in zip(*dtw_list)]

    return {
        "dtw": dtw_avg,
        "smape": smape_avg,
        "mae": mae_avg,
        "average": equi_avg,
        "stdev": equi_stdev
        # "time_taken" and "time_taken_50" can be added separately
    }

recorded_data = {}
for model in eval_list[0].keys():
    recorded_data[model] = summarize_eval_list(eval_list, model)


# Create a DataFrame with your data
data = {
    'LSTM': recorded_data["lstm"]["mae"],
    'N-BEATS': recorded_data["nbeats"]["mae"],
    'TFT': recorded_data["tft"]["mae"],
    'NHITS': recorded_data["nhits"]["mae"],
    'DLinear': recorded_data["dlinear"]["mae"],
    'NLinear': recorded_data["nlinear"]["mae"],
    'TSMixer': recorded_data["tsmixer"]["mae"],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate absolute values
abs_df = df.abs()

# Compute mean and standard deviation for each model
means = abs_df.mean(axis=0)
stds = abs_df.std(axis=0)

# Define unique colors for each model
models = means.index.tolist()

# Plot the bar chart with error bars
fig, ax = plt.subplots(figsize=(10, 4))

custom_colors = ["#c0c0c0", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#800080", "#D3D3D3", "#00FFFF"]

bars = ax.bar(means.index, means.values, capsize=5, color=custom_colors[:len(models)], alpha=0.8)
# Add labels on top of the bars with 2 decimal points

#c0c0c0

# Add title and labels
#ax.set_xlabel('Model', fontsize=24)
ax.set_ylabel('Mean MAE', fontsize=24)

ax.set_ylim(0.1, 0.2)

# Customize ticks
ax.tick_params(axis='x', labelsize=24, rotation=45)
ax.tick_params(axis='y', labelsize=24)

# Add grid lines on y-axis
ax.yaxis.grid(True, linestyle='-', alpha=0.7)
ax.xaxis.grid(True, linestyle='-', alpha=0.7)

# Add values on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,  # Slightly above the bar
        f'{height:.2f}',  # Format to 2 decimal points
        ha='center', va='bottom', fontsize=24
    )

# Add error bars
#for bar, mean, std in zip(bars, means.values, stds.values):
#    ax.text(
#        bar.get_x() + bar.get_width() / 2,
#        bar.get_height() + 0.1,
#        f'{mean:.2f}\n±{std:.2f}',
#        ha='center', va='bottom', fontsize=10
#    )

# Remove top and right spines (box)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Adjust layout and display
plt.tight_layout()
plt.savefig("mae.png")
plt.show()
