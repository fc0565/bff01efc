import os
import shutil
import pwlf
import zipfile

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
        "lstm_25": ("lstm_25", RNNModel),
        "lstm_50": ("lstm_50", RNNModel),
        "lstm_100": ("lstm_100", RNNModel),
        "lstm_250": ("lstm_250", RNNModel),
        "nlinear": ("nlinear", NLinearModel),
        "nlinear_25": ("nlinear_25", NLinearModel),
        "nlinear_50": ("nlinear_50", NLinearModel),
        "nlinear_100": ("nlinear_100", NLinearModel),
        "nlinear_250": ("nlinear_250", NLinearModel),
        "dlinear": ("dlinear", DLinearModel),
        "dlinear_25": ("dlinear_25", DLinearModel),
        "dlinear_50": ("dlinear_50", DLinearModel),
        "dlinear_100": ("dlinear_100", DLinearModel),
        "dlinear_250": ("dlinear_250", DLinearModel),
        "nhits": ("nhits", NHiTSModel),
        "nhits_25": ("nhits_25", NHiTSModel),
        "nhits_50": ("nhits_50", NHiTSModel),
        "nhits_100": ("nhits_100", NHiTSModel),
        "nhits_250": ("nhits_250", NHiTSModel),
        "nbeats": ("nbeats", NBEATSModel),
        "nbeats_25": ("nbeats_25", NBEATSModel),
        "nbeats_50": ("nbeats_50", NBEATSModel),
        "nbeats_100": ("nbeats_100", NBEATSModel),
        "nbeats_250": ("nbeats_250", NBEATSModel),
        "tft": ("tft", TFTModel),
        "tft_25": ("tft_25", TFTModel),
        "tft_50": ("tft_50", TFTModel),
        "tft_100": ("tft_100", TFTModel),
        "tft_250": ("tft_250", TFTModel),
        "tsmixer": ("tsmixer", TSMixerModel),
        "tsmixer_25": ("tsmixer_25", TSMixerModel),
        "tsmixer_50": ("tsmixer_50", TSMixerModel),
        "tsmixer_100": ("tsmixer_100", TSMixerModel),
        "tsmixer_250": ("tsmixer_250", TSMixerModel)
    }

    name = name.lower()
    if name not in model_mapping:
        print("Model not found")
        return None

    subfolder, model_class = model_mapping[name]
    model_path_full = f"{model_path}/{subfolder}_model.pth"

    return model_class.load(model_path_full)


# Function to process nested ZIP files
def process_zips(zip_path):
    dataframes = []  # List to store dataframes
    with zipfile.ZipFile(zip_path, 'r') as zfile:
        # Finding all CSV files in the ZIP
        csv_files = [f for f in zfile.namelist() if f.endswith('.csv')]
        for csv_file in csv_files:
            with zfile.open(csv_file) as file:
                df = pd.read_csv(file, index_col='timestamp')
                df.index = pd.to_datetime(df.index)
                dataframes.append(df)  # Append dataframe to the list
    return dataframes


def find_stable_point(series, model):
    window_length = 50
    polyorder = 2
    
    # Smooth the values (example with Savitzky-Golay filter)
    #smoothed_values = savgol_filter(series.values, window_length=window_length, polyorder=polyorder)
    smoothed_values = series.values
    
    # Convert the timestamp index to numeric (seconds since epoch)
    numeric_index = series.index.astype('int64') // 10**9  # Convert datetime to epoch seconds

    # Fit a two-segment piecewise linear model
    model = pwlf.PiecewiseLinFit(numeric_index, smoothed_values)
    breaks = model.fit(2)  # Specify 2 segments

    values = []
    for item in breaks:
        values.append(model.predict([item])[0])

    if abs( float(values[1] - values[0]) ) < 2:
        nick_point = breaks[0]
    else:
        nick_point = breaks[1]
    return nick_point





def regularize_data(data):
    most_common_interval = 100
    most_common_timedelta = pd.to_timedelta(most_common_interval, unit='s')
    regularized_data = data.resample(most_common_timedelta).bfill()
    return regularized_data


def eval_model(model, store_images=False, model_name="default", store_csv=False):
    """
    Notes:

    Altogether there are 33 datasets.
    We trained 25 datasets, and we evaluate the last 8 datasets here.
    Hence the range starts with 25.

    We forecast 460 data points after the first 36 points.
    Except for the last dataset, we shortened it (excluding unstable data).
    Hence, for the last dataset (index=32), we forecast 365 data points.
    """

    predict_length = 460

    res = {
        "equi": [],
        "smape": [],
        "mae": [],
        "dtw": []
    }

    result_path = f"result"
    # Define the folder path
    #folder_path = Path(result_path)

    # Remove the folder if it exists
    #if folder_path.exists():
    #    shutil.rmtree(folder_path)

    # Loop through the scaled_series_list from index 25 to the last one
    for i in range(25, len(scaled_series_list)):
        if i == 32:
            predict_length = 365
        # Get the first 36 values from each series (input values)
        first_36_values = scaled_series_list[i][:36]

        # Inverse transform the first 36 values back to the original scale
        
        #complete_original = scaler.inverse_transform(scaled_series_list[i]).pd_series().values
        complete_original = scaler.inverse_transform(scaled_series_list[i]).values().flatten()

        # Inverse transform the first 36 values back to the original scale
        #first_36_values_original = scaler.inverse_transform(first_36_values).pd_series().values
        first_36_values_original = scaler.inverse_transform(first_36_values).values().flatten()

        # Predict the next 460 values using the LSTM model
        scaled_prediction = model.predict(n=predict_length, series=first_36_values)

        # Inverse transform the predictions back to the original scale
        forecast_rnn_lstm = scaler.inverse_transform(scaled_prediction)

        # Get the original timestamps from the series
        original_timestamps = series[i].time_index  # Assuming series[i] is a Darts TimeSeries object

        # Create a DataFrame for the input values and predictions
        original_values_series = pd.Series(complete_original, index=original_timestamps)
        input_values_series = pd.Series(first_36_values_original, index=original_timestamps[:36])
        #predicted_values_series = pd.Series(forecast_rnn_lstm.pd_series().values, index=original_timestamps[36:36 + predict_length])
        predicted_values_series = pd.Series(forecast_rnn_lstm.values().flatten(), index=original_timestamps[36:36 + predict_length])

        # Combine both Series into a single DataFrame
        combined_series = pd.concat([input_values_series, predicted_values_series])

        # Similarity Metrics
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
            encoded_string = str(smape_value).replace('.', '_')  # Replace dots with underscores
            filename = f'result/{model_name}_predictions_{i}_{encoded_string}.png'
            plt.savefig(filename, format='png', dpi=300)
            plt.close()



        if store_csv:
            Path(result_path).mkdir(parents=True, exist_ok=True)
            filename = f'result/{model_name}_predictions_{i}.csv'
            combined_series.to_csv(filename)

    # Zip the folder
    #if store_images or store_csv:
    #    shutil.make_archive(folder_path, 'zip', folder_path)

    return res


def eval_model_moving(model):
    """

    Args:
      model:
      store_images:
      model_name:
      store_csv:

    Returns:

    """
    predict_length = 460

    res = {
        "equi": [],
        "smape": [],
        "mae": [],
        "dtw": []
    }

    DUT = 26

    #result_path = f"/content/{model_name}"
    # Define the folder path
    #folder_path = Path(result_path)

    # Remove the folder if it exists
    #if folder_path.exists():
    #    shutil.rmtree(folder_path)


    total_res = {}
    for time_step in range(0,250):
        my_time_step = time_step * 1

        # Loop through the scaled_series_list from index 25 to the last one
        for i in range(DUT, len(scaled_series_list)):
            if i == 32:
                predict_length = 365
            # Get the first 36 values from each series (input values)
            first_36_values = scaled_series_list[i][:36+my_time_step]

            # Inverse transform the first 36 values back to the original scale
            complete_original = scaler.inverse_transform(scaled_series_list[i]).pd_series().values

            # Inverse transform the first 36 values back to the original scale
            first_36_values_original = scaler.inverse_transform(first_36_values).pd_series().values

            # Predict the next 460 values using the LSTM model
            scaled_prediction = model.predict(n=predict_length, series=first_36_values)

            # Inverse transform the predictions back to the original scale
            forecast_rnn_lstm = scaler.inverse_transform(scaled_prediction)

            # Get the original timestamps from the series
            original_timestamps = series[i].time_index  # Assuming series[i] is a Darts TimeSeries object

            # Create a DataFrame for the input values and predictions
            original_values_series = pd.Series(complete_original, index=original_timestamps)
            input_values_series = pd.Series(first_36_values_original, index=original_timestamps[:36+my_time_step])
            predicted_values_series = pd.Series(forecast_rnn_lstm.pd_series().values, index=original_timestamps[36+my_time_step:36 + my_time_step + predict_length])

            # Combine both Series into a single DataFrame
            combined_series = pd.concat([input_values_series, predicted_values_series])

            # Similarity Metrics
            #smape_value = smape(scaled_series_list[i][36:36+predict_length], scaled_prediction)
            #mae_value = mae(scaled_series_list[i][36:36+predict_length], scaled_prediction)
            #dtw_value = dtw_metric(scaled_series_list[i][36:36+predict_length], scaled_prediction)
            equi = find_stable_point(combined_series, model)

            res["equi"].append(equi)
            #res["smape"].append(smape_value)
            #res["mae"].append(mae_value)
            #res["dtw"].append(dtw_value)

            #if store_images:
            #    Path(result_path).mkdir(parents=True, exist_ok=True)
            #    original_values_series[:36 + predict_length].plot(label='actual')
            #    predicted_values_series.plot(label=model_name, lw=3)
            #    plt.legend()
            #    encoded_string = str(smape_value).replace('.', '_')  # Replace dots with underscores
            #    filename = f'/content/drive/MyDrive/res/{model_name}_predictions_{i}_{encoded_string}.png'
            #    plt.savefig(filename, format='png', dpi=300)
            #    plt.close()
            #if store_csv:
            #    Path(result_path).mkdir(parents=True, exist_ok=True)
            #    filename = f'/content/drive/MyDrive/res/{model_name}_predictions_{i}.csv'
            #    combined_series.to_csv(filename)
            break
        total_res[my_time_step] = res

    # Zip the folder
    #if store_images or store_csv:
    #    shutil.make_archive(folder_path, 'zip', folder_path)

    return total_res


### 50 runs
recorded_data = {
    "tsmixer": {
        "time_taken": 2.263566,
        "time_taken_50": 113.178299,
        "dtw": [0.11718559840994827, 0.06259589534434837, 0.054275710298862204, 0.06320276990187643, 0.06400809892494194, 0.04950448651827823, 0.06791469603085702, 0.06998269994307016],
        "smape": [49.587684590219105, 32.55385555006749, 31.30020898256904, 29.516720236758072, 35.24207106384753, 43.077309387407965, 26.518107705054938, 20.86788392606509],
        "mae": [0.2813923771193623, 0.12149883473442849, 0.16413043629555882, 0.16360371486433298, 0.16122049816630032, 0.10994198428282739, 0.09785895018663787, 0.10936090460451907],
        "average": [1699888998.316127, 1701988359.034526, 1707243054.8849146, 1708960737.3229716, 1715622479.8328304, 1718055806.3901958, 1718135779.8031664, 1718202798.131023],
        "stdev": [1.9518906454878509, 7.770538687158782, 4.980550096908342, 5.500670476205568, 13.378067382282161, 11.95472420519925, 10.086517540730252, 1.9952426309008184]
    },
    "tft": {
        "time_taken": 2.390071,
        "time_taken_50": 119.503546,
        "dtw": [0.05817098937915786, 0.07623671419554194, 0.05902845400874186, 0.08597812823380062, 0.061669683106570054, 0.05419500677518981, 0.07351614562172057, 0.06000814719081519],
        "smape": [31.545133383342606, 38.80779605042831, 29.78849815641643, 60.65321310477225, 30.61520895568559, 53.78048544453501, 46.399987688672105, 26.630862354007686],
        "mae": [0.1564964890088477, 0.1787686284638348, 0.15490662643124864, 0.27896261151937585, 0.14245644112396968, 0.14799206113905003, 0.21261687761772755, 0.13666304048681815],
        "average": [1699907495.9153905, 1701997288.079208, 1707243517.5585103, 1708960673.7710512, 1715611487.1644142, 1718030691.8482134, 1718125201.985682, 1718215507.860463], 
        "stdev": [21.58055860934432, 98.24633870049055, 19.281252418390704, 9.905126124589668, 12.906273111060122, 1.0610596008834976, 11797.387379908268, 18580.091796710898]
    },
    "nbeats": {
        "time_taken": 5.178114,
        "time_taken_50": 258.905686,
        "dtw": [0.04622322592226231, 0.05386272819713856, 0.05386072851409753, 0.06588481841150651, 0.05662601738443591, 0.06933898379226906, 0.061740963934893374, 0.054954183034400035],
        "smape": [22.344141945815192, 30.108864862837144, 32.645159791342685, 35.939339853396305, 24.793201367535783, 69.83324612414476, 38.420787993295974, 22.711036030337972],
        "mae": [0.10139956399814692, 0.12487857191243275, 0.16846431069098103, 0.19244763953285057, 0.12272708036033056, 0.22692063256504097, 0.16063201459383905, 0.11760450108288752],
        "average": [1699902872.5779355, 1701983616.240522, 1707242271.4605796, 1708956693.5882058, 1715611402.9252627, 1718030407.448772, 1718132684.094876, 1718210187.0426636],
        "stdev": [8.924005441866347, 20.743198760247765, 10.297934751146432, 7.469059489041648, 11.747304663708409, 0.5745374682510707, 18.14847675953888, 10486.873766419998]
    },
    "nhits": {
        "time_taken": 3.074327,
        "time_taken_50": 153.716332,
        "dtw": [0.06205175625425914, 0.06832408118869919, 0.052548461587335935, 0.06806586840069552, 0.0648969064590533, 0.1039181713059607, 0.07340167640845258, 0.07591035986674598],
        "smape": [27.490963351952608, 30.479493815508707, 13.765925152864083, 37.630601914789665, 24.395704489925045, 71.1074325341548, 41.12110587116107, 19.302267580308033],
        "mae": [0.12838295392072369, 0.12593134275686085, 0.08186092720688462, 0.20153942614834383, 0.12034392406098546, 0.227444856774961, 0.1737766980059993, 0.09913255250943288],
        "average": [1699905576.7590187, 1701982676.9621234, 1707242225.996207, 1708958746.735854, 1715617911.1006567, 1718030432.3012228, 1718124838.506436, 1718215112.9766638],
        "stdev": [28.547962925745015, 17.079410170670513, 13.421135697869264, 14.103981508245022, 25.588770649907477, 0.2607929529053078, 7835.760435218384, 8859.537148765883]
    },
    "dlinear": {
        "time_taken": 2.334155,
        "time_taken_50": 116.707729,
        "dtw": [0.09167731810574287, 0.07314937515718342, 0.05009411278995241, 0.060017987898494995, 0.07212702982556181, 0.06206748631586496, 0.10595753064659076, 0.06498063810943586],
        "smape": [43.75604554191403, 34.57481041097469, 15.83860733126142, 15.781424813791189, 21.201949018362377, 54.16357058767303, 53.14335816924282, 19.52267566257844],
        "mae": [0.235047189979963, 0.149274209907288, 0.09657959429103308, 0.09573110599471395, 0.10701589152995636, 0.14386180755038888, 0.25200220866155704, 0.10044249332391103],
        "average": [1699906323.1326668, 1701986385.2255344, 1707245377.417461, 1708962850.2760546, 1715603767.8791773, 1718030491.5369573, 1718116449.9633567, 1718198600.1501257],
        "stdev": [27.721662060640547, 35.13531523783683, 32.40288161078879, 18.17164206239685, 9069.258892921158, 0.30240064830806135, 0.13874664863432928, 0.26351510935172395]
   },
    "nlinear": {
        "time_taken": 2.268800,
        "time_taken_50": 113.439986,
        "dtw": [0.09430282191564006, 0.07297924611965394, 0.05058771509932725, 0.06229020156886166, 0.07217846262819294, 0.05842242572927793, 0.10581375467547957, 0.06740915573016647],
        "smape": [41.58796465409368, 32.710727068500525, 14.624293086041984, 15.817678218646236, 20.96179895317476, 51.95320400947684, 51.36482052095743, 19.45450719390397],
        "mae": [0.21912884094827698, 0.13845402066649404, 0.0881450837699381, 0.09595946240774544, 0.10548452991688424, 0.1345600441703378, 0.23934862376187405, 0.09988856147418534],
        "average": [1699904498.345445, 1701986127.6580832, 1707243673.4543326, 1708960063.9074423, 1715604101.2460868, 1718030499.8730433, 1718116445.767283, 1718211248.4504347],
        "stdev": [14.874474394675195, 28.490694104650714, 17.039488602441853, 13.106777508174757, 8455.31782140846, 0.29483981302296264, 0.104827228590078, 10002.644532081118]
    },
    "lstm": {
        "time_taken": 2.856834,
        "time_taken_50": 142.841676,
        "dtw": [0.33325983855503355, 0.07478149492319627, 0.12638200662557902, 0.07417389580329596, 0.08342571986019828, 0.04470214928448538, 0.10368183014764588, 0.06514904826638189],
        "smape": [61.86388958270863, 31.485850051007606, 34.45605860659376, 21.853842925885427, 20.761020995196834, 41.66796683860885, 51.33819880485284, 20.289320195579702],
        "mae": [0.39138332330808817, 0.13141041199428297, 0.24401431522202913, 0.1409202132817049, 0.1043662146924708, 0.09458679759746066, 0.23945284591761676, 0.1039665672715226],
        "average": [1699904727.3475082, 1701983495.9591649, 1707245045.8820968, 1708941000.0, 1715599026.6569045, 1718034633.8148215, 1718116456.8232071, 1718203563.7644932],
        "stdev": [11.052721220105555, 46.881367520372436, 16.125237769400908, 0.0, 1.382234504956981, 2.0327349065636873, 0.1558264300246118, 3760.589770737665]
    }
}

# Plot time difference

ref_equi = get_ref_stable_point()
ref_equi_list = []
for i in range(25,33):
    # Convert to pandas datetime object
    dt_object = pd.to_datetime(ref_equi[str(i)])
    # Convert to epoch time
    epoch_time = int(dt_object.timestamp())
    ref_equi_list.append(epoch_time)

ref_numeric = pd.to_numeric(ref_equi_list, errors='coerce')
final_ref_equi = pd.DataFrame(ref_equi_list, columns=['date/time ref'])


final_time_container = []
final_time_container.append(final_ref_equi)

for item in recorded_data:
    #my_numeric = pd.to_numeric(eval_results[item]["equi"], errors='coerce')
    if item == "lstm":
        my_numeric = recorded_data[item]['average']
    elif item == "nlinear":
        my_numeric = recorded_data[item]['average']
    elif item == "dlinear":
        my_numeric = recorded_data[item]['average']
    elif item == "nhits":
        my_numeric = recorded_data[item]['average']
    elif item == "nbeats":
        my_numeric = recorded_data[item]['average']
    elif item == "tft":
        my_numeric = recorded_data[item]['average']
    elif item == "tsmixer":
        my_numeric = recorded_data[item]['average']
    else:
        my_numeric = pd.to_numeric(eval_results[item]["equi"], errors='coerce')
    #print(my_numeric)
    #print(type(my_numeric))
    #print(ref_numeric)
    my_points = [ ((x - y)/3600.0) for x, y in zip(my_numeric, ref_numeric)]
    my_column_name = f"{item}_time_diff"
    final_time_container.append(pd.DataFrame(my_points, columns=[my_column_name]))

final_time_diff = pd.concat(final_time_container, axis=1)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

custom_colors = ["#c0c0c0", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#800080", "#D3D3D3", "#00FFFF"]

# Example Data
data = {
    'LSTM': final_time_diff["lstm_time_diff"].to_list(),
    'N-BEATS': final_time_diff["nbeats_time_diff"].to_list(),
    'TFT': final_time_diff["tft_time_diff"].to_list(),
    'NHits': final_time_diff["nhits_time_diff"].to_list(),
    'DLinear': final_time_diff["dlinear_time_diff"].to_list(),
    'NLinear': final_time_diff["nlinear_time_diff"].to_list(),
    'TSMixer': final_time_diff["tsmixer_time_diff"].to_list()
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate absolute values
abs_df = df.abs()

# Compute mean for each model
means = abs_df.mean(axis=0)

# Define unique colors for each model
models = means.index.tolist()

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 4))

#custom_colors = ["#c0c0c0", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#800080", "#D3D3D3", "#00FFFF"]

# Create the bar chart
bars = ax.bar(models, means, color=custom_colors[:len(models)], alpha=0.8)

# Add labels on top of the bars with 2 decimal points
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # Center the text horizontally
        height,  # Position text above the bar
        f'{height:.2f}',  # Format with 2 decimal points
        ha='center',  # Horizontal alignment
        va='bottom',  # Vertical alignment
        fontsize=18  # Font size
    )

# Add labels and title
#ax.set_xlabel('Models', fontsize=18)
ax.set_ylabel('RMSE Time Diff. (hr)', fontsize=18)
#ax.set_title('Mean Time Difference Across Models', fontsize=20)

# Customize ticks
ax.tick_params(axis='x', labelsize=18, rotation=45)
ax.tick_params(axis='y', labelsize=18)

# Adjust layout and save
plt.tight_layout()
plt.savefig("fig_7a.png")
#plt.show()