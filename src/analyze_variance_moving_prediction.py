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


lstm_moving_results = [np.float64(-2.657834061384201), np.float64(-2.23080886873934), np.float64(-3.6276397913694383), np.float64(-3.7927262782388262), np.float64(-3.0077344413598377), np.float64(-1.4603754291931788), np.float64(-4.034413304328918), np.float64(-4.125504648950365), np.float64(-4.954643443028132), np.float64(-4.902815460496479), np.float64(-4.784910395079189), np.float64(-4.402915954391162), np.float64(-4.17483167734411), np.float64(-4.14617149664296), np.float64(-3.100062128239208), np.float64(-4.6263194164302615), np.float64(-5.480693000488811), np.float64(-5.489814074701733), np.float64(-5.545953252712885), np.float64(-5.485549796952141), np.float64(-5.5235231573051875), np.float64(-5.525327479110824), np.float64(-5.525924543407228), np.float64(-5.442419141001172), np.float64(-5.527622986568345), np.float64(-5.386415028174718), np.float64(-5.358964628179868), np.float64(-5.3866974143849475), np.float64(-5.385772451758385), np.float64(-5.527075661818187), np.float64(-5.387188238634003), np.float64(-5.386722544895278), np.float64(-5.386901721821891), np.float64(-5.38639211555322), np.float64(-5.386389105187522), np.float64(-5.3595218877659905), np.float64(-5.081484002272288), np.float64(-5.080808247261577), np.float64(-5.114139152699047), np.float64(-5.135681457122167), np.float64(-5.136405735413233), np.float64(-5.377306705382135), np.float64(-5.35939141260253), np.float64(-5.136392850544717), np.float64(-5.3824510839912625), np.float64(-5.136025286316872), np.float64(-5.136219436923663), np.float64(-5.08143713010682), np.float64(-4.776772534052531), np.float64(-5.098206894463963), np.float64(-5.1090405787362), np.float64(-5.136794422136413), np.float64(-5.136275018188688), np.float64(-5.364593884216415), np.float64(-5.134677369859483), np.float64(-5.135490180187755), np.float64(-5.362956136398846), np.float64(-5.133323811954922), np.float64(-5.35962825761901), np.float64(-5.137923448350694), np.float64(-5.383270867334472), np.float64(-5.367246705624792), np.float64(-5.386388889153799), np.float64(-5.381097959611151), np.float64(-5.3633339775933155), np.float64(-5.360236986014578), np.float64(-5.138700643380483), np.float64(-5.13652703444163), np.float64(-5.361738565829065), np.float64(-5.379765269027816), np.float64(-5.368776572876507), np.float64(-5.385805521011353), np.float64(-5.386388892200258), np.float64(-5.380319373475181), np.float64(-5.119300048881107), np.float64(-5.136388650668992), np.float64(-5.106370019581583), np.float64(-5.1074864584207536), np.float64(-5.087107629709774), np.float64(-5.081749080287086), np.float64(-5.0867766930659615), np.float64(-5.108101631535424), np.float64(-5.088891510698531), np.float64(-3.9687598503960504), np.float64(-2.7833869791693155), np.float64(-3.970199653837416), np.float64(-3.969724243150817), np.float64(-3.9682905149459837), np.float64(-3.9674796371327505), np.float64(-3.6825635155704286), np.float64(-3.9169565477636126), np.float64(-4.021380965974596), np.float64(-5.081218668752246), np.float64(-3.6673781434694925), np.float64(-3.971387046906683), np.float64(-3.970161014066802), np.float64(-3.67340648955769), np.float64(-3.9678622861703237), np.float64(-3.4174420807758965), np.float64(-3.6988022116157744), np.float64(-3.968820943236351), np.float64(-3.2761970499489044), np.float64(-3.2348047519392438), np.float64(-3.3589572257465785), np.float64(-3.3574494442012575), np.float64(-3.414507185882992), np.float64(-3.969940228793356), np.float64(-5.081114297707876), np.float64(-5.081678523222606), np.float64(-5.08151029712624), np.float64(-5.085967762536472), np.float64(-5.085560577180651), np.float64(-5.08587280717161), np.float64(-3.971740031308598), np.float64(-4.026766195562151), np.float64(-3.4203865603605905), np.float64(-3.9314130200942357), np.float64(-3.956246479286088), np.float64(-3.9618590819173387), np.float64(-2.775277946194013), np.float64(-3.360614755352338), np.float64(-2.8867481864823237), np.float64(-2.88714449485143), np.float64(-3.494847768611378), np.float64(-3.6787050947878095), np.float64(-3.9719257051414916), np.float64(-2.8875320421324835), np.float64(-3.776617198255327), np.float64(-2.887593091196484), np.float64(-2.8888351988130148), np.float64(-3.4247123738129934), np.float64(-3.9000206680430307), np.float64(-3.9689576325151656), np.float64(-3.9848972558312945), np.float64(-4.025297673543294), np.float64(-3.9732471840911443), np.float64(-3.9241940599017675), np.float64(-2.8863889701498877), np.float64(-2.24971525894271), np.float64(-2.778877729707294), np.float64(-2.886390138996972), np.float64(-3.4147238663832344), np.float64(-2.7818252111143535), np.float64(-2.2756819397211077), np.float64(-2.886583353016112), np.float64(-2.8905347631374996), np.float64(-3.4132130459282135), np.float64(-3.684088216225306), np.float64(-3.959719807571835), np.float64(-4.009707788560125), np.float64(-4.0284709488021), np.float64(-4.025746391614278), np.float64(-3.934693877167172), np.float64(-3.4141122829914092), np.float64(-2.7762329247925015), np.float64(-2.775341020292706), np.float64(-2.887533153626654), np.float64(-3.4133633869224123), np.float64(-2.8868470348914466), np.float64(-3.4952648103237154), np.float64(-3.415170246031549), np.float64(-3.416732693447007), np.float64(-3.690066904094484), np.float64(-3.6635866814851763), np.float64(-3.957295094397333), np.float64(-3.898437756167518), np.float64(-3.423375001417266), np.float64(-3.4110980054405), np.float64(-2.8868489235639574), np.float64(-2.8873868254820505), np.float64(-2.779755242731836), np.float64(-1.8061552340454525), np.float64(-1.8170649616585837), np.float64(-1.3337588698996439), np.float64(-2.7949158701631758), np.float64(-3.4282280147737927), np.float64(-3.9553240958187317), np.float64(-4.026747251881494), np.float64(-3.9704106546110576), np.float64(-3.915516070259942), np.float64(-3.7008021765947343), np.float64(-3.955536428226365), np.float64(-3.691353919042481), np.float64(-3.415121821761131), np.float64(-3.3683290341827603), np.float64(-3.4167889874511297), np.float64(-3.6882569012376996), np.float64(-3.968624705804719), np.float64(-4.024597726662954), np.float64(-4.02387720088164), np.float64(-4.0312226887544), np.float64(-4.022642567091518), np.float64(-4.013323151336776), np.float64(-4.025007737477621), np.float64(-4.033580259548293), np.float64(-4.0277807017829685), np.float64(-3.96983832206991), np.float64(-3.9746051493618224), np.float64(-3.972397598955366), np.float64(-3.9686815570460428), np.float64(-3.954230403833919), np.float64(-3.9695719385809367), np.float64(-3.9510670903656218), np.float64(-3.910772428512573), np.float64(-3.896192431582345), np.float64(-3.415874242120319), np.float64(-2.88859991437859), np.float64(-2.7778764766454698), np.float64(-0.23681643936369154), np.float64(0.7099357834789488), np.float64(-0.01587593721018897), np.float64(-2.77802034093274), np.float64(-0.3103046119875378), np.float64(-2.7775041219923233), np.float64(-0.13622557189729478), np.float64(0.07090566138426463), np.float64(0.10786700950728523), np.float64(0.1044883241918352), np.float64(0.12465132978227403), np.float64(1.4002752847803963), np.float64(1.480530432595147), np.float64(1.031747326519754), np.float64(0.34552733706103433), np.float64(0.0015368053648206923), np.float64(-0.0020265305042266846), np.float64(0.0016915206114451091), np.float64(0.23125004331270854), np.float64(0.0019808894395828246), np.float64(-0.01493997057278951), np.float64(-0.003930640750461154), np.float64(-0.0014868524339463976), np.float64(0.0019953624407450356), np.float64(0.0009788618485132854), np.float64(-0.0017264018456141154), np.float64(0.42151331682999926), np.float64(0.50233705315325), np.float64(0.0017621344327926635), np.float64(-0.004035155110889011), np.float64(0.0019112024704615275), np.float64(0.0019178185860315958), np.float64(-0.005017126599947612), np.float64(-0.13388323214319017), np.float64(-0.2775533008575439), np.float64(-0.18974422646893396), np.float64(-0.3494454099734624), np.float64(-2.779115352431933), np.float64(-2.775449864069621), np.float64(-2.7788764541678956), np.float64(-2.7798447234100765), np.float64(-2.78011288245519)]
dlinear_moving_results = [np.float64(-1.8565760091940562), np.float64(-1.2258812435468038), np.float64(-1.8860199279255336), np.float64(-1.1353548011514876), np.float64(-0.9369023388624191), np.float64(-1.069267239305708), np.float64(-0.8944811211691962), np.float64(-1.8846287718746397), np.float64(-1.7246392279863358), np.float64(-1.8991428468624751), np.float64(-0.9418265848027335), np.float64(-1.7813908909426794), np.float64(-0.7817590759860145), np.float64(-1.265564315120379), np.float64(-0.25172875112957427), np.float64(-1.8587998096810447), np.float64(-5.636886079046461), np.float64(-5.636770260267787), np.float64(-5.592789348761241), np.float64(-5.635870195560985), np.float64(-5.580925597945849), np.float64(-5.589849420587222), np.float64(-5.58095606525739), np.float64(-5.580838189456198), np.float64(-5.636464648842812), np.float64(-5.528485816518466), np.float64(-5.3592894855472775), np.float64(-5.580450721912914), np.float64(-5.525750291016367), np.float64(-5.612446650266647), np.float64(-5.584623105194834), np.float64(-5.526449785762363), np.float64(-5.580740275912815), np.float64(-5.526487402982182), np.float64(-5.525472244289186), np.float64(-5.386388912200927), np.float64(-5.082175451450878), np.float64(-5.083721961643961), np.float64(-5.131770469877455), np.float64(-5.136391614344385), np.float64(-5.386550293564796), np.float64(-5.525298578408029), np.float64(-5.525287307898203), np.float64(-5.387319601641761), np.float64(-5.52699020312892), np.float64(-5.386388889683618), np.float64(-5.362835409906175), np.float64(-5.08188780327638), np.float64(-4.774194969468646), np.float64(-5.081118328968684), np.float64(-5.135910444392098), np.float64(-5.525277858376503), np.float64(-5.386396660076247), np.float64(-5.525696955256992), np.float64(-5.38648400982221), np.float64(-5.382872457106908), np.float64(-5.387825690905253), np.float64(-5.386406144897143), np.float64(-5.388244097630183), np.float64(-5.386399870316188), np.float64(-5.52784234603246), np.float64(-5.386396554178662), np.float64(-5.525453899635209), np.float64(-5.5809216586086485), np.float64(-5.385778468185001), np.float64(-5.386474504272143), np.float64(-5.371457068059179), np.float64(-5.137687656018469), np.float64(-5.386391330626275), np.float64(-5.3867587615384), np.float64(-5.387628610928854), np.float64(-5.386888841986656), np.float64(-5.52528129849169), np.float64(-5.136427599853939), np.float64(-5.109095530377494), np.float64(-5.114038992060555), np.float64(-5.085295869178242), np.float64(-5.081348163021936), np.float64(-4.049711100127962), np.float64(-4.02891889459557), np.float64(-4.80751670592361), np.float64(-5.0846929033597315), np.float64(-5.0829098093509675), np.float64(-4.02963404973348), np.float64(-5.096017401284642), np.float64(-5.080838297075696), np.float64(-5.082744868265258), np.float64(-4.025855381025209), np.float64(-4.0460981505446965), np.float64(-3.9700329548120497), np.float64(-4.025883470972379), np.float64(-5.081550400588248), np.float64(-5.08144417696529), np.float64(-4.031685905125406), np.float64(-4.807784484889773), np.float64(-5.082455633944935), np.float64(-3.970217812458674), np.float64(-4.026305942667856), np.float64(-3.968931541774008), np.float64(-3.963224677907096), np.float64(-4.025458831787109), np.float64(-3.914245802693897), np.float64(-3.6687311936749354), np.float64(-3.929413520561324), np.float64(-3.954499905771679), np.float64(-3.9702491550975374), np.float64(-4.024649788339933), np.float64(-4.809827261037297), np.float64(-5.08186607129044), np.float64(-4.791482039822473), np.float64(-5.084351354108916), np.float64(-5.0858105295234255), np.float64(-4.05247562110424), np.float64(-4.026273681985008), np.float64(-4.026285233630074), np.float64(-3.9284900959332782), np.float64(-3.9726439538266924), np.float64(-3.9713868540525437), np.float64(-3.9700189393758776), np.float64(-3.415186311999957), np.float64(-3.6650115648243164), np.float64(-3.6737011600202982), np.float64(-3.4954573334587944), np.float64(-3.967775927649604), np.float64(-3.967313559651375), np.float64(-4.025707183149126), np.float64(-3.666556541919708), np.float64(-3.969739865263303), np.float64(-3.6873216064108743), np.float64(-3.4957874947124057), np.float64(-3.9672611343198354), np.float64(-3.9619414375887976), np.float64(-3.972378678454293), np.float64(-4.026011951035923), np.float64(-4.0290023042758305), np.float64(-4.023542494045364), np.float64(-3.972925473385387), np.float64(-3.678421191043324), np.float64(-3.3612892541620467), np.float64(-3.672120102710194), np.float64(-3.774924779401885), np.float64(-3.922616309920947), np.float64(-3.6629052858220206), np.float64(-3.4159381786982217), np.float64(-3.6746828480561575), np.float64(-3.7770981489949755), np.float64(-3.9660088543097176), np.float64(-3.967834997905625), np.float64(-3.970857115652826), np.float64(-4.02574864546458), np.float64(-4.025281256768438), np.float64(-4.02942369957765), np.float64(-3.9699151653051374), np.float64(-3.6912995003991655), np.float64(-2.88655318028397), np.float64(-2.889357518619961), np.float64(-3.414778172373772), np.float64(-3.686954209142261), np.float64(-3.666086337765058), np.float64(-3.775510149995486), np.float64(-3.9492360423670876), np.float64(-3.690424713227484), np.float64(-3.9596443854437933), np.float64(-3.9280702613459693), np.float64(-3.9685739518536463), np.float64(-3.9638588655657236), np.float64(-3.9071530783838697), np.float64(-3.687728146976895), np.float64(-3.4067111315992142), np.float64(-3.41924159500334), np.float64(-3.4157715610663097), np.float64(-2.8887911437617406), np.float64(-2.88795883986685), np.float64(-2.7767194341950945), np.float64(-3.414433793624242), np.float64(-3.9138663223716947), np.float64(-3.967862015300327), np.float64(-3.9709445366594527), np.float64(-3.9723386624786587), np.float64(-3.9585923351844152), np.float64(-3.953828686144617), np.float64(-3.97055095911026), np.float64(-3.917063548233774), np.float64(-3.674162095983823), np.float64(-3.415003080897861), np.float64(-3.674038249784046), np.float64(-3.9193035017119513), np.float64(-3.9607194974687365), np.float64(-3.9725182149145337), np.float64(-3.9720921135611005), np.float64(-4.021848268906275), np.float64(-3.9711941101153694), np.float64(-3.970952258043819), np.float64(-3.972007459799449), np.float64(-4.022279928194152), np.float64(-3.9701975118451647), np.float64(-3.960418544742796), np.float64(-3.9634833328591452), np.float64(-3.9591040905978945), np.float64(-3.9661509223116767), np.float64(-3.921147813399633), np.float64(-3.9565321462684206), np.float64(-3.91972847017977), np.float64(-3.7745952767795985), np.float64(-3.7744838257630664), np.float64(-3.4166724150710635), np.float64(-3.3590676584508685), np.float64(-3.4071748973263634), np.float64(-3.3597503936953013), np.float64(-2.8870981199873818), np.float64(-3.4142204086648094), np.float64(-3.410786810517311), np.float64(-2.8872741054164037), np.float64(-2.899109421902233), np.float64(-2.777732214861446), np.float64(-0.10357538905408648), np.float64(-0.10839488453335232), np.float64(-0.10828556206491258), np.float64(-0.27580206063058643), np.float64(0.6654241578446494), np.float64(0.13909102658430736), np.float64(0.0010960563023885092), np.float64(0.027812763386302525), np.float64(-0.302894949581888), np.float64(-0.2965562454197142), np.float64(-0.12736116164260441), np.float64(-0.09074888832039303), np.float64(-0.31212614549530876), np.float64(-0.14046166565683152), np.float64(-0.30165534926785365), np.float64(-0.31422619111008115), np.float64(-0.3340336255894767), np.float64(-0.359502469168769), np.float64(-0.19222857117652892), np.float64(-0.10738456812169817), np.float64(0.0017684754398134019), np.float64(-0.13320290909873114), np.float64(-2.279245795143975), np.float64(-0.1382362527317471), np.float64(-0.13762837211290996), np.float64(-2.2784437972307203), np.float64(-2.7756354471047717), np.float64(-2.7869117506345114), np.float64(-2.7762253754668764), np.float64(-2.8850253854195276), np.float64(-2.886456460091803), np.float64(-2.888596018486553), np.float64(-2.888081973923577), np.float64(-2.8955917473634085), np.float64(-2.8887661078241136)]
nlinear_moving_results = [np.float64(-1.9363961146937476), np.float64(-1.8729627604617012), np.float64(-1.9796630648771922), np.float64(-1.861468295984798), np.float64(-1.8919552399714787), np.float64(-1.8789512681298786), np.float64(-1.9753869471947352), np.float64(-2.0181719452142715), np.float64(-2.0585676082637576), np.float64(-2.04978680756357), np.float64(-1.8875390658775966), np.float64(-1.8673576802015304), np.float64(-1.7021496180031035), np.float64(-1.748326519727707), np.float64(-1.1536240244574016), np.float64(-1.873015718460083), np.float64(-5.169230339593358), np.float64(-5.478855474193891), np.float64(-5.45895180536641), np.float64(-5.423842012484869), np.float64(-5.474607805344793), np.float64(-5.525313159889645), np.float64(-5.525278205076853), np.float64(-5.306513563460774), np.float64(-5.525299666391478), np.float64(-5.378370324638155), np.float64(-5.342035090592172), np.float64(-5.386106018688944), np.float64(-5.358648944099744), np.float64(-5.528850905034277), np.float64(-5.38640925473637), np.float64(-5.386534350514412), np.float64(-5.386774817374017), np.float64(-5.38673129969173), np.float64(-5.360228152275085), np.float64(-5.108993543320232), np.float64(-5.060063066813681), np.float64(-5.080833333995607), np.float64(-5.083503600292736), np.float64(-5.081477072371377), np.float64(-5.127234947019153), np.float64(-5.386568755904833), np.float64(-5.137499285340309), np.float64(-5.12802865392632), np.float64(-5.386626939045058), np.float64(-5.108049471113417), np.float64(-5.082321506076389), np.float64(-4.775450147920185), np.float64(-4.735765153037177), np.float64(-4.776313781672054), np.float64(-5.08893389251497), np.float64(-5.138227466079924), np.float64(-5.108643961482578), np.float64(-5.387184761961302), np.float64(-5.108587486280335), np.float64(-5.109554456671079), np.float64(-5.127625885274675), np.float64(-5.109162109825347), np.float64(-5.129714719984267), np.float64(-5.112051455709669), np.float64(-5.384202000167635), np.float64(-5.360008318490452), np.float64(-5.386970341669189), np.float64(-5.3865596279170775), np.float64(-5.115535513758659), np.float64(-5.12982488963339), np.float64(-5.091006646553676), np.float64(-5.107448691858186), np.float64(-5.113774060209592), np.float64(-5.1360213119453855), np.float64(-5.117696449028121), np.float64(-5.359029401275847), np.float64(-5.361246069272359), np.float64(-5.108457349141439), np.float64(-4.045699386795362), np.float64(-5.084162975880835), np.float64(-3.9698216083314684), np.float64(-3.9750098500649136), np.float64(-3.9687937190797595), np.float64(-3.9598012522856396), np.float64(-3.9702338523334926), np.float64(-4.025421514776018), np.float64(-4.025358204974069), np.float64(-3.9703934658235975), np.float64(-4.026617075337304), np.float64(-4.024859061307377), np.float64(-3.9697345155477524), np.float64(-3.7757899845971004), np.float64(-3.9635046335723665), np.float64(-3.6638778044117823), np.float64(-3.9585219319661458), np.float64(-4.024963379767206), np.float64(-4.040862264169587), np.float64(-3.9289752551582127), np.float64(-3.9699583082728918), np.float64(-3.970152969956398), np.float64(-3.6649391765726937), np.float64(-3.9137715672122106), np.float64(-3.415730220410559), np.float64(-3.665091260539161), np.float64(-3.920490498079194), np.float64(-3.3596111791001424), np.float64(-3.271042971412341), np.float64(-3.4163489681482315), np.float64(-3.412587163448334), np.float64(-3.4981723097960153), np.float64(-3.918180331985156), np.float64(-4.021772092911932), np.float64(-4.025692381262779), np.float64(-4.0254344063335), np.float64(-4.026461270650228), np.float64(-4.025584131280581), np.float64(-3.9744033444590037), np.float64(-3.7753066364924113), np.float64(-3.9686494517326354), np.float64(-3.4149972352054383), np.float64(-3.682244902253151), np.float64(-3.6814693385362625), np.float64(-3.77555815882153), np.float64(-2.776098262468974), np.float64(-2.8867617377969954), np.float64(-2.8895978277259404), np.float64(-2.886538360450003), np.float64(-3.4210752718316186), np.float64(-3.6644831962717905), np.float64(-3.914890574945344), np.float64(-3.358037490447362), np.float64(-3.6663817146751616), np.float64(-2.886429942846298), np.float64(-2.8866545838779873), np.float64(-3.4193516623311573), np.float64(-3.6687551080518297), np.float64(-3.898810190690888), np.float64(-3.952007912397385), np.float64(-3.9697702573405373), np.float64(-3.9297895658016206), np.float64(-3.6927375508679283), np.float64(-2.886904094947709), np.float64(-2.77537092520131), np.float64(-2.8875601610210206), np.float64(-3.4149301621649), np.float64(-3.415605559878879), np.float64(-2.888356804980172), np.float64(-2.777790097925398), np.float64(-3.3580345355802113), np.float64(-3.4169060418340895), np.float64(-3.666269586549865), np.float64(-3.8859724128246307), np.float64(-3.9241157501935957), np.float64(-3.9692414116197163), np.float64(-3.9707183521323732), np.float64(-3.9652262861861125), np.float64(-3.690040404531691), np.float64(-3.3591112577252917), np.float64(-1.8148585086398654), np.float64(-1.8212626253234017), np.float64(-2.8870249880022474), np.float64(-3.4135157442092896), np.float64(-2.888135552075174), np.float64(-3.4234221993552314), np.float64(-3.420994069178899), np.float64(-3.416953811711735), np.float64(-3.6768813639879228), np.float64(-3.6672128947575886), np.float64(-3.8936809787485336), np.float64(-3.6799995024336707), np.float64(-3.423538136879603), np.float64(-3.4131680226325987), np.float64(-2.8884751400020385), np.float64(-2.889700639843941), np.float64(-2.8897131909264457), np.float64(-2.2787358044253456), np.float64(-2.2784550952911378), np.float64(-1.3263635760545731), np.float64(-2.886830129822095), np.float64(-3.41537356475989), np.float64(-3.7753424820635053), np.float64(-3.9509532366858586), np.float64(-3.91266091770596), np.float64(-3.6901197760634954), np.float64(-3.668695507778062), np.float64(-3.701734975443946), np.float64(-3.665758002665308), np.float64(-3.41329748014609), np.float64(-3.276139963666598), np.float64(-3.4196845779154037), np.float64(-3.669674271941185), np.float64(-3.8954193118545746), np.float64(-3.924382467534807), np.float64(-3.930158236953947), np.float64(-3.9697891676425936), np.float64(-3.9637815103265974), np.float64(-3.922437025109927), np.float64(-3.9203297875987158), np.float64(-3.969883135822084), np.float64(-3.9563634682363933), np.float64(-3.8969969455401103), np.float64(-3.9053966230816313), np.float64(-3.6895410838392046), np.float64(-3.691612519621849), np.float64(-3.6783057576417924), np.float64(-3.775415678554111), np.float64(-3.6750076179371938), np.float64(-3.493810023797883), np.float64(-3.4978091028001574), np.float64(-3.3599114027288226), np.float64(-2.7966842730177772), np.float64(-2.7831318424807656), np.float64(-2.7842193902201124), np.float64(-0.02404606448279487), np.float64(-2.886311467819744), np.float64(-2.8837103904618155), np.float64(-0.19236844374073878), np.float64(-2.778678862916099), np.float64(0.0034763919644885594), np.float64(0.33943010641468896), np.float64(0.4002126803663042), np.float64(0.46447853558593327), np.float64(0.15280719565020667), np.float64(1.0727153329716788), np.float64(0.9068561976485782), np.float64(0.7008466200033824), np.float64(0.5586755808856753), np.float64(0.0023833292722702025), np.float64(-0.10517619801892175), np.float64(0.23808407816621993), np.float64(0.37973900781737435), np.float64(-0.09441917565133837), np.float64(-0.0013498371839523314), np.float64(-0.09658748394913144), np.float64(-0.07323957780996958), np.float64(-0.13596727993753221), np.float64(-0.13154447423087226), np.float64(0.0024651604228549534), np.float64(0.4195284680525462), np.float64(0.48838674969143336), np.float64(-0.0001313863197962443), np.float64(-0.27528627687030366), np.float64(-0.06772852659225465), np.float64(-0.08773273673322465), np.float64(-0.2773747116327286), np.float64(-2.280383614897728), np.float64(-2.776164586544037), np.float64(-2.278726722266939), np.float64(-2.776590067214436), np.float64(-2.777156930565834), np.float64(-2.7786914663182363), np.float64(-2.8871290220154657), np.float64(-2.7798017009099323), np.float64(-2.780073215166728)]
nhits_moving_results = [np.float64(-2.894716338713964), np.float64(-4.185394333998362), np.float64(-2.6956392637226316), np.float64(-2.674293913576338), np.float64(-3.813812156981892), np.float64(-2.802618299590217), np.float64(-4.65215355667803), np.float64(-2.6038009748856226), np.float64(-1.5543355835146375), np.float64(-1.6464279433091482), np.float64(-0.7712171798944474), np.float64(0.2507723109589683), np.float64(-1.4623920161856545), np.float64(-1.841712630589803), np.float64(-1.3690330029858484), np.float64(-2.5089789017041526), np.float64(-5.636989778478941), np.float64(-5.904910776151551), np.float64(-5.580870881941584), np.float64(-5.637048724492391), np.float64(-5.636518551442358), np.float64(-5.860111152331035), np.float64(-5.63638895491759), np.float64(-5.637515474292967), np.float64(-5.63639523545901), np.float64(-5.636448680228657), np.float64(-5.636388889418708), np.float64(-5.62070250093937), np.float64(-5.636415100561248), np.float64(-5.636524500979317), np.float64(-5.636403968930244), np.float64(-5.636843383577135), np.float64(-5.640601383248965), np.float64(-5.636858195000225), np.float64(-5.636090592212147), np.float64(-5.581061649719874), np.float64(-5.526559956007533), np.float64(-5.581535420484013), np.float64(-5.581228754454189), np.float64(-5.581128391755952), np.float64(-5.608770629763603), np.float64(-5.579157497353024), np.float64(-5.5812040491898856), np.float64(-5.580940798984633), np.float64(-5.636251727872424), np.float64(-5.5268985725773705), np.float64(-5.581931299169859), np.float64(-5.5255068156454294), np.float64(-5.525277853276995), np.float64(-5.135324389470948), np.float64(-5.389099741048283), np.float64(-4.014673406547971), np.float64(-5.119968221916093), np.float64(-5.361115227606561), np.float64(-5.525328421526485), np.float64(-5.132141916155815), np.float64(-5.582178586920103), np.float64(-3.932512932618459), np.float64(-5.528652309510443), np.float64(-5.118848383227984), np.float64(-5.525295995804998), np.float64(-5.137228979667028), np.float64(-5.386808786922031), np.float64(-5.358690391845173), np.float64(-5.386823906964726), np.float64(-5.386538428001933), np.float64(-5.386830653614468), np.float64(-5.1345375088850655), np.float64(-5.108645841876665), np.float64(-5.387885972791248), np.float64(-5.138597085343467), np.float64(-5.525615415374438), np.float64(-5.370488047334883), np.float64(-5.386519961820708), np.float64(-5.386055440306664), np.float64(-5.386388890345891), np.float64(-5.104800141983562), np.float64(-5.109818975792991), np.float64(-5.111194120115704), np.float64(-5.081946025888125), np.float64(-5.080838253961669), np.float64(-5.1087785430749255), np.float64(-5.0826807739999555), np.float64(-5.083290283944872), np.float64(-4.02528289967113), np.float64(-4.0261654653814105), np.float64(-3.9659564274549486), np.float64(-3.0398032177819148), np.float64(-1.8013094741768307), np.float64(-3.618384181857109), np.float64(-3.972012068165673), np.float64(-2.5125467320283255), np.float64(-2.736329858435525), np.float64(-3.970026640163528), np.float64(-3.9699157351255416), np.float64(-3.975210845536656), np.float64(-3.4253464213344786), np.float64(-3.497529336810112), np.float64(-3.419794825381703), np.float64(-3.9704116612010534), np.float64(-3.9487111431360247), np.float64(-3.6822354627317853), np.float64(-3.4957632189326815), np.float64(-3.8992788051234353), np.float64(-3.413275213771396), np.float64(-3.4965650377670925), np.float64(-3.9720924441019694), np.float64(-3.97147096991539), np.float64(-3.7732406527466242), np.float64(-4.028521787722905), np.float64(-4.025871526002884), np.float64(-3.9702205774519177), np.float64(-3.9702621986468634), np.float64(-4.0231957421037885), np.float64(-3.778248208032714), np.float64(-4.025507555603981), np.float64(-4.025280453695191), np.float64(-4.028214922613568), np.float64(-3.969809277984831), np.float64(-3.9696960581673517), np.float64(-4.026500648127662), np.float64(-3.9702351256873873), np.float64(-3.969722359312905), np.float64(-4.025281741751565), np.float64(-4.024810278746817), np.float64(-4.033684291044871), np.float64(-4.022941725651423), np.float64(-4.025309681230121), np.float64(-3.964771218829685), np.float64(-3.9732200574874876), np.float64(-3.97310358941555), np.float64(-3.972336667511198), np.float64(-4.025646532509062), np.float64(-3.971461466352145), np.float64(-4.026674344009823), np.float64(-3.9718645256757736), np.float64(-3.6926878734429676), np.float64(-3.9703882946570714), np.float64(-3.6655646177132923), np.float64(-3.8992474910285737), np.float64(-3.3590307115183937), np.float64(-4.019673364493582), np.float64(-3.276332633694013), np.float64(-3.970251579284668), np.float64(-3.4300041021903356), np.float64(-3.9685417073965072), np.float64(-3.682105816801389), np.float64(-3.970444942249192), np.float64(-3.912215677830908), np.float64(-3.970370252132416), np.float64(-3.9136072457498976), np.float64(-3.7842275648646884), np.float64(-4.016619505153762), np.float64(-3.691690626806683), np.float64(-2.888802625272009), np.float64(-3.4156828649838764), np.float64(-3.4978972178035312), np.float64(-3.6732916979657277), np.float64(-3.4988000707493887), np.float64(-3.416980798509386), np.float64(-3.918247256875038), np.float64(-3.3597117078966563), np.float64(-3.773944197760688), np.float64(-3.6799547083510293), np.float64(-3.936465187801255), np.float64(-3.9082957214779324), np.float64(-3.6627693204747307), np.float64(-0.5177595905462901), np.float64(-3.414112778570917), np.float64(-3.659637602236536), np.float64(-3.414804471267594), np.float64(-3.4241687999169033), np.float64(-3.41488536424107), np.float64(-2.7776842774947483), np.float64(-3.495635418958134), np.float64(-3.6903824987014135), np.float64(-3.9723972092734443), np.float64(-3.9154335419336954), np.float64(-3.9736468225717543), np.float64(-3.952516994542546), np.float64(-3.9307327421506244), np.float64(-3.7775620034005906), np.float64(-3.688315640621715), np.float64(-3.4331921623150508), np.float64(-3.6871695101261137), np.float64(-3.910707333286603), np.float64(-3.773800134725041), np.float64(-3.9591917269097436), np.float64(-3.9523058503203923), np.float64(-3.4380710054768455), np.float64(-3.9655353778600695), np.float64(-3.9146912683380974), np.float64(-3.967642224232356), np.float64(-3.678458064728313), np.float64(-3.9745505477322474), np.float64(-3.676367709901598), np.float64(-3.783463974263933), np.float64(-3.69063548207283), np.float64(-3.9595620639456643), np.float64(-3.9025316066874396), np.float64(-3.66350757141908), np.float64(-3.7774495112233692), np.float64(-3.6697232325871787), np.float64(-3.6898956376976435), np.float64(-3.41072100089656), np.float64(-3.665344951285256), np.float64(1.267000478439861), np.float64(-3.9290504668156307), np.float64(-3.417540640367402), np.float64(-3.4966209330161413), np.float64(-2.905023172563977), np.float64(-3.4151133629348545), np.float64(-0.322519499791993), np.float64(-2.887028112411499), np.float64(0.44408491790294646), np.float64(1.7577234041028553), np.float64(0.13067944705486298), np.float64(-0.13431877824995253), np.float64(0.4870987478892008), np.float64(-0.19241810977458954), np.float64(-2.777278356750806), np.float64(0.001057682302263048), np.float64(-0.29308752914269764), np.float64(1.5976503137747446), np.float64(-0.13772195484903124), np.float64(0.3797407822476493), np.float64(0.367499044669999), np.float64(-0.3320338154501385), np.float64(-2.7768407925632266), np.float64(-0.1193403865231408), np.float64(0.42694612476560806), np.float64(9.890682167477078e-05), np.float64(-0.13450066169102987), np.float64(0.7124838477373123), np.float64(0.001794758505291409), np.float64(9.852025243971083e-05), np.float64(-0.10135302868154314), np.float64(-0.0007984962728288438), np.float64(-0.34787541965643565), np.float64(-0.13079884396659003), np.float64(-0.13322469135125478), np.float64(-2.7757822597026824), np.float64(-2.7898254956801734), np.float64(-2.7756527953015433), np.float64(-2.889550365077125), np.float64(-2.7854657214217715), np.float64(-2.7812454011705188), np.float64(-2.882661235133807), np.float64(-3.282891274624401), np.float64(-2.779218710925844)]
nbeats_moving_results = [np.float64(-2.6349784348408383), np.float64(-4.943090400232209), np.float64(-4.635252154535717), np.float64(-3.546186786558893), np.float64(-5.68809642944071), np.float64(-4.949444458617104), np.float64(-2.1092778348260457), np.float64(-3.9872711341248617), np.float64(-2.396981278326776), np.float64(2.072640498942799), np.float64(-4.896344428658486), np.float64(-3.833643820418252), np.float64(-3.1739796998765732), np.float64(-3.9426518895228706), np.float64(-5.131407558917999), np.float64(-4.72630430314276), np.float64(-3.0745800149440767), np.float64(-5.8613265795177885), np.float64(-5.636900614367591), np.float64(-5.910348504318131), np.float64(-5.082094943390953), np.float64(-5.584856564733717), np.float64(-5.623059953716066), np.float64(-5.526213476061821), np.float64(-5.636793339186244), np.float64(-5.388765831192335), np.float64(-5.042463811834653), np.float64(-4.777325272096528), np.float64(-5.590065574182405), np.float64(-5.580833362340927), np.float64(-5.5265087519751654), np.float64(-5.526008970340093), np.float64(-5.580972781313791), np.float64(-5.5269198974635865), np.float64(-5.633113348351585), np.float64(-5.525279801819059), np.float64(-5.3865276406870946), np.float64(-5.135223093893793), np.float64(-4.969925899373161), np.float64(-5.136235098308987), np.float64(-5.581630510025554), np.float64(-5.386640062199699), np.float64(-5.3830761066410275), np.float64(-5.580536066293717), np.float64(-5.525310567220052), np.float64(-5.386349334451888), np.float64(-5.386389151745372), np.float64(-5.088777270913124), np.float64(-5.088179438577758), np.float64(-5.119040235016081), np.float64(-4.805905415150854), np.float64(-5.386014299260245), np.float64(-5.3817028931776685), np.float64(-5.5788163426187305), np.float64(-5.137386491894722), np.float64(-5.525363166530927), np.float64(-5.377333779798613), np.float64(-5.580485136442714), np.float64(-5.526158227192031), np.float64(-5.3864053101009794), np.float64(-5.386542181306415), np.float64(-5.525761276019944), np.float64(-5.580833970838123), np.float64(-5.118021254671945), np.float64(-5.372401901019944), np.float64(-5.517171018520991), np.float64(-5.389500911831856), np.float64(-5.525680524375703), np.float64(-5.108871713942952), np.float64(-5.387082225614124), np.float64(-5.527342968185743), np.float64(-5.136135733789867), np.float64(-5.3865004036161634), np.float64(-5.106730807953411), np.float64(-5.386678872638279), np.float64(-5.092559760610262), np.float64(-5.139588041437997), np.float64(-5.11218563079834), np.float64(-4.777174451218711), np.float64(-4.048330080972778), np.float64(-5.358705264859729), np.float64(-5.114391537838512), np.float64(-5.37601224164168), np.float64(-5.12202757484383), np.float64(-5.113765780064795), np.float64(-4.019595822493235), np.float64(-5.087717320455445), np.float64(-4.0250521488984425), np.float64(-3.9701799055602813), np.float64(-3.61166522734695), np.float64(-3.497571574780676), np.float64(-3.1777975410223007), np.float64(-4.022055698103375), np.float64(-3.4382385567824048), np.float64(-5.085497080418799), np.float64(-3.386969874633683), np.float64(-3.4142467502090668), np.float64(-3.966530619263649), np.float64(-3.1940277361869813), np.float64(-3.8915684410598543), np.float64(-4.023505251672533), np.float64(-3.921418065296279), np.float64(-3.2774345142973793), np.float64(-3.9659628511137432), np.float64(-2.7279301738739012), np.float64(-3.2757264193561344), np.float64(-3.970278608666526), np.float64(-3.9188498501645195), np.float64(-3.6774333910809625), np.float64(-3.969460638629066), np.float64(-4.045604178441895), np.float64(-3.9459581875138814), np.float64(-3.6650437804725433), np.float64(-2.5582712139023673), np.float64(-2.288686580657959), np.float64(-3.9147081747319965), np.float64(-3.9669067399369347), np.float64(-2.9312498832411236), np.float64(-3.6720705966817007), np.float64(-3.4252303879790835), np.float64(-4.019409675399462), np.float64(-3.4181842984755835), np.float64(-3.4165955636236403), np.float64(-4.023812808858024), np.float64(-4.024889259537061), np.float64(-4.024234097136391), np.float64(-3.6479985945092306), np.float64(-3.958417894111739), np.float64(-4.046324877010451), np.float64(-2.891667077541351), np.float64(-4.023455566830105), np.float64(-3.9713730855782825), np.float64(-3.916451880865627), np.float64(-4.050260496603118), np.float64(-4.049786912467745), np.float64(-5.082507364551226), np.float64(-3.970294123225742), np.float64(-3.9814975656403435), np.float64(-3.905360976325141), np.float64(-5.081680813696649), np.float64(-3.953981025616328), np.float64(-5.097178927527533), np.float64(-5.080840794973903), np.float64(-3.9589407208893035), np.float64(-4.025009504225519), np.float64(-3.699225981036822), np.float64(-3.688142862253719), np.float64(-4.023843887580766), np.float64(-4.780326073037253), np.float64(-4.027599587837855), np.float64(-4.049659264948633), np.float64(-4.051869150863753), np.float64(-3.974806998173396), np.float64(-3.669177910486857), np.float64(-3.664567304915852), np.float64(-3.9123988594611485), np.float64(-3.6872996365361743), np.float64(-4.026340130766233), np.float64(-3.773459249006377), np.float64(-4.0243565913703705), np.float64(-4.025568512148327), np.float64(-3.415568809244368), np.float64(-3.4174065612422098), np.float64(-3.416978598766857), np.float64(-3.9714950474765565), np.float64(-4.018851817117797), np.float64(-4.0259341263108785), np.float64(-2.8862844714853497), np.float64(-2.8865684654977586), np.float64(-3.69181527753671), np.float64(-2.8867303013139303), np.float64(-3.7751020156012642), np.float64(-3.914695089260737), np.float64(-3.3647710597515106), np.float64(-3.4162463542487886), np.float64(-3.480626538462109), np.float64(-3.9106817480590608), np.float64(-3.97319646080335), np.float64(-3.415383087661531), np.float64(-3.9003514241509967), np.float64(-3.9705344441864225), np.float64(-3.9750732189416884), np.float64(-3.6731624019145968), np.float64(-4.023410724401474), np.float64(-3.38013680530919), np.float64(-3.91203227672312), np.float64(-3.699274884727266), np.float64(-3.7735088949071036), np.float64(-3.9120229287279975), np.float64(-3.9713324003749424), np.float64(-4.050105706916915), np.float64(-3.956273296541638), np.float64(-4.021158393290308), np.float64(-3.7757305154535508), np.float64(-4.019679735501607), np.float64(-3.6655186878972583), np.float64(-3.9725017691320845), np.float64(-3.9732635990778604), np.float64(-3.667269333137406), np.float64(-3.9659790290064283), np.float64(-3.89259590572781), np.float64(-3.950386045045323), np.float64(-3.690600112213029), np.float64(-3.6908946563800176), np.float64(-3.6929972750610776), np.float64(-3.967147630519337), np.float64(-3.6930512811077967), np.float64(-3.901669303642379), np.float64(-3.9060362413856717), np.float64(-3.4155288399590384), np.float64(-3.95518319606781), np.float64(-2.887058099243376), np.float64(-0.3280976210037867), np.float64(-3.411897018816736), np.float64(2.030737834705247), np.float64(0.8099257561895582), np.float64(-0.10823981649345822), np.float64(0.7107034966018465), np.float64(-2.78086218310727), np.float64(-0.08854536314805349), np.float64(2.211517349349128), np.float64(0.6108490983645122), np.float64(1.9821459213230346), np.float64(-0.09235788881778717), np.float64(0.7035754070017073), np.float64(1.3872673404879041), np.float64(2.8625390640232298), np.float64(0.6579405807124243), np.float64(0.0011928519937727186), np.float64(-2.779082118603918), np.float64(0.5898835447761748), np.float64(1.7201624362336265), np.float64(-0.10825806531641219), np.float64(-0.0022040502892600165), np.float64(1.704875641266505), np.float64(0.0008933558728959826), np.float64(-2.779775870243708), np.float64(-0.2786970411406623), np.float64(-0.002103226449754503), np.float64(1.8169447035921944), np.float64(-0.38376838637722865), np.float64(-0.10738472594155206), np.float64(-2.7764050600263808), np.float64(-2.781467639870114), np.float64(-2.8891693529817792), np.float64(-2.7778253260585997), np.float64(-2.778312176995807), np.float64(-2.8901633471250534), np.float64(-3.3467160211669076), np.float64(-2.778722592194875)]
tft_moving_results = [np.float64(-1.8640242183870739), np.float64(2.104177134368155), np.float64(-3.181557531952858), np.float64(0.38326051387521953), np.float64(-5.625842523707284), np.float64(-6.194110316303041), np.float64(-3.151609272758166), np.float64(-5.324637174871233), np.float64(-1.3850706043508318), np.float64(-5.875184551344978), np.float64(-2.7765151162279977), np.float64(-6.031524685886171), np.float64(-4.744548979666498), np.float64(-4.696392268538475), np.float64(-3.573153732419014), np.float64(-5.860509104000197), np.float64(5.890908854272631), np.float64(-5.641342311369049), np.float64(-2.941228156818284), np.float64(-5.888850922120942), np.float64(-5.635965485903952), np.float64(-5.663749141163296), np.float64(-3.1089191314909193), np.float64(-5.41743738717503), np.float64(-3.633310823970371), np.float64(-5.326975128120846), np.float64(-5.026311667097939), np.float64(-5.222834494378832), np.float64(-4.830776098767917), np.float64(-4.999697725110583), np.float64(-5.218639919559161), np.float64(-4.757435205115212), np.float64(-4.747538810835945), np.float64(-5.120898860957888), np.float64(-4.910836087995105), np.float64(-4.632330357631048), np.float64(-5.080839878718058), np.float64(-5.081025529901186), np.float64(-5.523434290885925), np.float64(-3.8470025763909024), np.float64(-5.099673504564497), np.float64(-3.20596285449134), np.float64(-5.08628351688385), np.float64(-5.382166516184807), np.float64(-5.386963054140409), np.float64(-5.387245700028208), np.float64(-1.13721130364471), np.float64(-5.376394639280107), np.float64(-4.775287122527758), np.float64(-5.530040280752712), np.float64(-5.08605708056026), np.float64(-5.363523859447903), np.float64(-5.082315286397934), np.float64(-5.389804854061868), np.float64(-5.581147331396739), np.float64(-5.389692083332274), np.float64(-5.08573787106408), np.float64(-5.386027808785439), np.float64(-2.886466112865342), np.float64(-5.103007260958353), np.float64(-5.10782811039024), np.float64(-5.086519262724453), np.float64(-4.081709829105272), np.float64(-5.528069988489151), np.float64(-5.084995111227036), np.float64(-5.113932657705413), np.float64(-3.498359341290262), np.float64(-2.2237131590313384), np.float64(-5.084240691728062), np.float64(-5.525979831947221), np.float64(-5.088728911413087), np.float64(-5.1153030504783), np.float64(-5.137047273185518), np.float64(-5.101648494071431), np.float64(-5.086451037393676), np.float64(-5.112461129426956), np.float64(-5.102251348098119), np.float64(-3.275592230492168), np.float64(-3.060887877676222), np.float64(-3.901784380012088), np.float64(-3.9680439417892033), np.float64(-4.029935913615756), np.float64(2.909218512972196), np.float64(-3.7874254695574443), np.float64(-5.107637942102221), np.float64(-5.082727533976237), np.float64(-5.084403330286344), np.float64(-4.034059544735484), np.float64(-0.048375897341304354), np.float64(-3.9689206488927207), np.float64(-5.081646307905515), np.float64(-4.026203684210778), np.float64(-5.087389501267009), np.float64(-3.6896267202827664), np.float64(-2.9878431520197126), np.float64(-3.045414226584964), np.float64(-3.4155082219176824), np.float64(-3.969924430648486), np.float64(0.04785251008139716), np.float64(-3.422255578769578), np.float64(-2.971951582630475), np.float64(-2.9125921349393), np.float64(-3.1645097522603143), np.float64(-3.6655362057023577), np.float64(-1.7786696290307575), np.float64(-3.2752880891164144), np.float64(-3.3552413694063823), np.float64(-3.0181750726699828), np.float64(-3.6809520641962687), np.float64(-3.41525162630611), np.float64(-3.421257090303633), np.float64(-3.8985000569952857), np.float64(-3.961523232791159), np.float64(-3.6661784849564234), np.float64(-3.973721212744713), np.float64(-3.3404110827710896), np.float64(-3.415668014023039), np.float64(-3.275628589193026), np.float64(-3.689723625249333), np.float64(-2.881129714449247), np.float64(-3.67625344965193), np.float64(-3.66822219034036), np.float64(-3.4157580556472142), np.float64(-3.9620499252610735), np.float64(-3.6692797664139007), np.float64(-3.9701249503427083), np.float64(-3.96972808904118), np.float64(-3.668992147511906), np.float64(-2.783161368171374), np.float64(-3.4120730433199142), np.float64(-2.7826299932930203), np.float64(-3.6646226153771084), np.float64(-3.6843811841805776), np.float64(-3.4173562468422785), np.float64(-3.414666615459654), np.float64(-3.684579683608479), np.float64(-3.6795151880714627), np.float64(-3.446038218670421), np.float64(-2.77899867243237), np.float64(-2.883507547577222), np.float64(-3.2771812985340754), np.float64(-3.416071650385857), np.float64(-2.2707870393329195), np.float64(-2.0317272209458883), np.float64(-2.2756787414020963), np.float64(-2.25225283033318), np.float64(-2.279629565808508), np.float64(-3.4132015442185932), np.float64(-3.4003560284111236), np.float64(-3.6678226050403384), np.float64(-3.6682072746753693), np.float64(-3.422160249153773), np.float64(-2.8918388958772026), np.float64(-3.417357801662551), np.float64(-3.353015966878997), np.float64(-2.8865393656492233), np.float64(-2.2734766484631432), np.float64(-2.7779535874393253), np.float64(-1.4666627162032657), np.float64(-2.7775282254483966), np.float64(-2.7761912203497356), np.float64(-2.886388916571935), np.float64(-2.8870775214831035), np.float64(-2.775307447049353), np.float64(-2.88644773847527), np.float64(-3.3563566482729383), np.float64(-3.496503051916758), np.float64(-3.412620485358768), np.float64(-2.7760018030140134), np.float64(-2.882262570924229), np.float64(-3.671258660554886), np.float64(-1.3376093720727498), np.float64(-3.4082522817452747), np.float64(-3.4961424687835905), np.float64(-3.49139213833544), np.float64(-3.4161541317568886), np.float64(-2.8868994172414144), np.float64(-3.4948944309684964), np.float64(-3.6735865430037182), np.float64(-3.9769060843520694), np.float64(-3.7792984331978694), np.float64(-3.6793319449822106), np.float64(-3.7771861014101242), np.float64(-3.4168411089976627), np.float64(-2.8892053627305563), np.float64(-2.8941004290183385), np.float64(-3.6612815776136185), np.float64(-3.414379858639505), np.float64(-3.41607937766446), np.float64(-3.687769900229242), np.float64(-3.410952555073632), np.float64(-3.6802002391550275), np.float64(-3.7007379622591867), np.float64(-3.6786735966470507), np.float64(-3.960420450501972), np.float64(-3.965186551478174), np.float64(-3.95360941529274), np.float64(-3.6835671187109416), np.float64(-3.9624500243531333), np.float64(-4.015262403753069), np.float64(-3.676351951493157), np.float64(-3.687605493929651), np.float64(-3.981183936794599), np.float64(-3.9362983669175042), np.float64(-3.664908037185669), np.float64(-3.9586158825953803), np.float64(-3.9366419102748234), np.float64(-3.665902399553193), np.float64(-3.414983392159144), np.float64(-2.8788974973890515), np.float64(2.5689938704172772), np.float64(4.676815499862035), np.float64(4.054453164405293), np.float64(3.618722115423944), np.float64(2.251385154856576), np.float64(2.813236564397812), np.float64(2.1798571381303997), np.float64(1.7282291577259699), np.float64(3.34607110063235), np.float64(3.2015427017211913), np.float64(1.5304432124561733), np.float64(3.724347141318851), np.float64(0.8933802431159549), np.float64(2.007484644651413), np.float64(0.2455639798111386), np.float64(1.4487927501731448), np.float64(0.32787044339709814), np.float64(0.5300331106450823), np.float64(0.564377189874649), np.float64(0.7883204758829541), np.float64(2.110250495539771), np.float64(2.0077936398983), np.float64(1.2526177571217219), np.float64(1.2282647803094653), np.float64(1.0025736254453659), np.float64(1.0467298829555511), np.float64(1.0877806922462252), np.float64(1.5666767649518119), np.float64(0.8178856698671977), np.float64(2.274317740334405), np.float64(1.8731614427434073), np.float64(2.308586889637841), np.float64(2.3764065683550304), np.float64(2.3730353056059945), np.float64(4.166516404019462), np.float64(2.7539179642995197), np.float64(1.9805962403615316), np.float64(1.066435983048545), np.float64(2.278461535639233), np.float64(-3.4013351426521936)]
tsmixer_moving_results = [np.float64(-1.310023953649733), np.float64(-1.2043671612607108), np.float64(-1.2134287240770127), np.float64(-1.1599390306075414), np.float64(-0.7830209545294444), np.float64(-0.7038670100106134), np.float64(-0.9907851911915673), np.float64(-0.7659723071257273), np.float64(-1.3743769538402557), np.float64(-1.5462441278166241), np.float64(-1.7932943704393174), np.float64(-1.5670795887708664), np.float64(-1.2053939057058758), np.float64(-1.0683810291025373), np.float64(-0.2858783296081755), np.float64(-0.18025350166691675), np.float64(0.4774117798275418), np.float64(0.4872571864393022), np.float64(0.7139731329017215), np.float64(-0.1959137502643797), np.float64(0.6028044576777353), np.float64(0.1855174207025104), np.float64(-0.0126449939277437), np.float64(0.017141786813735963), np.float64(0.08356094413333469), np.float64(-0.0504817318254047), np.float64(0.6126280245516035), np.float64(1.0570028731558059), np.float64(0.8827858993079927), np.float64(1.3320834358533225), np.float64(0.8282652299933964), np.float64(0.7052820246749454), np.float64(-0.5336771129237281), np.float64(1.1466157429085837), np.float64(0.8684857195615768), np.float64(-5.907836177282864), np.float64(-5.609300366308954), np.float64(-5.636388888955116), np.float64(-5.635027069184515), np.float64(-5.635620289908515), np.float64(-5.636670953631401), np.float64(-5.8623734444379805), np.float64(-6.190525052017636), np.float64(-6.191944454312324), np.float64(-7.053055555555556), np.float64(-6.192479370435079), np.float64(-6.026160973972744), np.float64(-5.9147344744867745), np.float64(-5.636401583022542), np.float64(-5.89089442147149), np.float64(-6.1921249759859505), np.float64(-6.192130524781015), np.float64(-6.025321668982506), np.float64(-6.190454729596774), np.float64(-6.030310531324811), np.float64(-6.025610165463553), np.float64(-6.192157386276457), np.float64(-6.182846986651421), np.float64(-6.1912548544009525), np.float64(-6.02669543074237), np.float64(-6.028191392885314), np.float64(-5.914651071363025), np.float64(-6.191678791774644), np.float64(-6.0253466159105304), np.float64(-6.02646208471722), np.float64(-5.914598008328014), np.float64(-5.900798626807001), np.float64(-5.862039848764738), np.float64(-5.861219094395637), np.float64(-5.866723232534197), np.float64(-5.913504400584433), np.float64(-5.914415696197086), np.float64(-6.183103581269582), np.float64(-5.8729715825451745), np.float64(-5.636424953805076), np.float64(-5.656561750968297), np.float64(-5.636872035529878), np.float64(-5.647862468163172), np.float64(-5.634902831514676), np.float64(-5.637243282066451), np.float64(-5.63643947660923), np.float64(-5.631308757000499), np.float64(-5.625784590641658), np.float64(-5.580661811497476), np.float64(-5.581807764371236), np.float64(-5.581649054487547), np.float64(-5.580413961542977), np.float64(-5.52575316104624), np.float64(-5.581852334207959), np.float64(-5.52543624970648), np.float64(-5.52674033595456), np.float64(-5.581029409898652), np.float64(-5.581995385289193), np.float64(-5.52551745335261), np.float64(-5.529195241663191), np.float64(-5.388650296860271), np.float64(-5.387054238650534), np.float64(-5.386390286485354), np.float64(-5.3878071030643255), np.float64(-5.386632632944319), np.float64(-5.3863920805851615), np.float64(-5.384547027349472), np.float64(-5.386843361987008), np.float64(-5.386276499695248), np.float64(-5.38624262743526), np.float64(-5.384562468992339), np.float64(-5.386410885585679), np.float64(-5.3865916956133315), np.float64(-5.525927132897907), np.float64(-5.386388890345891), np.float64(-5.525281866987546), np.float64(-5.526360169053078), np.float64(-5.386465886698828), np.float64(-5.383066702816222), np.float64(-5.386394868824217), np.float64(-5.383912397821744), np.float64(-5.381796346041892), np.float64(-5.380279242859946), np.float64(-5.374043694535891), np.float64(-5.139972247746256), np.float64(-5.358774190809991), np.float64(-5.360785532924864), np.float64(-5.36306331468953), np.float64(-5.3863947126600475), np.float64(-5.38549399389161), np.float64(-5.3865523974763025), np.float64(-5.366749955349499), np.float64(-5.37141991270913), np.float64(-5.360963666505284), np.float64(-5.360345208777321), np.float64(-5.361560478806496), np.float64(-5.136511241065131), np.float64(-5.373191672099961), np.float64(-5.138637636105219), np.float64(-5.359536580774519), np.float64(-5.373862200578054), np.float64(-5.362702056633101), np.float64(-5.383653555048837), np.float64(-5.386400094628334), np.float64(-5.380754963755607), np.float64(-5.358871430357297), np.float64(-5.13741329756048), np.float64(-5.118781560129589), np.float64(-5.113538792530695), np.float64(-5.114691358738475), np.float64(-5.115074228114552), np.float64(-5.117270999948183), np.float64(-5.386689639356401), np.float64(-5.133718021445804), np.float64(-5.132382053203053), np.float64(-5.387480502194829), np.float64(-5.360128624770376), np.float64(-5.126619093219439), np.float64(-5.137505418989393), np.float64(-5.095891046590276), np.float64(-5.106089531315698), np.float64(-5.112930948270692), np.float64(-5.10940492120054), np.float64(-5.110448079307874), np.float64(-5.107918049494425), np.float64(-5.103859120938513), np.float64(-5.110609856711494), np.float64(-5.108295336498155), np.float64(-5.1079291854302085), np.float64(-5.111568009323544), np.float64(-5.107969520025783), np.float64(-5.103084254463513), np.float64(-5.108822699917687), np.float64(-5.093462170097563), np.float64(-5.097696675658226), np.float64(-5.104085710313585), np.float64(-5.08230010204845), np.float64(-5.081248889366786), np.float64(-5.082519702182876), np.float64(-5.108481968243916), np.float64(-5.108992132345835), np.float64(-5.108070038292143), np.float64(-5.108564692470763), np.float64(-5.106170299980375), np.float64(-5.107887181705899), np.float64(-5.102852064039972), np.float64(-5.103669340345594), np.float64(-5.08143534011311), np.float64(-5.088724642131064), np.float64(-5.0872432838545905), np.float64(-5.088852350446913), np.float64(-5.140544427037239), np.float64(-5.116708722975519), np.float64(-5.11110268579589), np.float64(-5.109625803960694), np.float64(-5.125026784009403), np.float64(-5.1227330751551525), np.float64(-5.1172079055839115), np.float64(-5.110225428740184), np.float64(-5.122480224503411), np.float64(-5.09908253437943), np.float64(-5.10871166255739), np.float64(-5.0912055957979625), np.float64(-5.104264263444477), np.float64(-5.109998201992776), np.float64(-5.110749748812782), np.float64(-5.1109897016154395), np.float64(-5.083338648610645), np.float64(-5.081537033518155), np.float64(-5.0836786118480894), np.float64(-5.086537520620558), np.float64(-5.081044949756729), np.float64(-4.025560970240169), np.float64(-3.7751177852021325), np.float64(-3.9696100552214517), np.float64(-5.134433059692383), np.float64(-5.082187956571579), np.float64(-5.083479528493352), np.float64(-5.084391022655699), np.float64(-5.076911798715591), np.float64(-5.089099616474575), np.float64(-5.109824604127142), np.float64(-4.807427577111456), np.float64(-5.105486034684711), np.float64(0.001529878642823961), np.float64(-0.1087727623515659), np.float64(-0.11125518004099529), np.float64(-3.6570374874273934), np.float64(-3.9710630028115377), np.float64(-3.9736474491490257), np.float64(-3.9604380181762906), np.float64(-3.9114915813340083), np.float64(-3.9572117584281497), np.float64(-3.968424643940396), np.float64(-3.9711885235044693), np.float64(-3.954082886113061), np.float64(-3.9659053646855886), np.float64(-3.9683835726976393), np.float64(-3.9255860355827545), np.float64(-3.774428353640768), np.float64(-3.683577268322309), np.float64(-3.6868666967418458), np.float64(-3.6748078810506395), np.float64(-3.664502629637718), np.float64(-3.6750181695487765), np.float64(-3.7750954818725586), np.float64(-3.943552798496352), np.float64(-3.969199425313208), np.float64(-3.965219873653518), np.float64(-3.9671400562259884), np.float64(-3.9781092720561557), np.float64(-3.967511338657803), np.float64(-4.025650448666679), np.float64(-3.971940739353498), np.float64(-3.964974809421433)]

import numpy as np
import matplotlib.pyplot as plt


# Example lists from your data
lstm_variance = np.var(np.abs(lstm_moving_results))
dlinear_variance = np.var(np.abs(dlinear_moving_results))
nlinear_variance = np.var(np.abs(nlinear_moving_results))
nhits_variance = np.var(np.abs(nhits_moving_results))
nbeats_variance = np.var(np.abs(nbeats_moving_results))
tft_variance = np.var(np.abs(tft_moving_results))
tsmixer_variance = np.var(np.abs(tsmixer_moving_results))


#lstm_variance = np.sqrt(np.mean(np.square(lstm_moving_results)))
#dlinear_variance = np.sqrt(np.mean(np.square(dlinear_moving_results)))
#nlinear_variance = np.sqrt(np.mean(np.square(nlinear_moving_results)))
#nhits_variance = np.sqrt(np.mean(np.square(nhits_moving_results)))
#nbeats_variance = np.sqrt(np.mean(np.square(nbeats_moving_results)))
#tft_variance = np.sqrt(np.mean(np.square(tft_moving_results)))
#tsmixer_variance = np.sqrt(np.mean(np.square(tsmixer_moving_results)))



# Create a dictionary of variances
variances = {
    "LSTM": lstm_variance,
    "NBeats": nbeats_variance,
    "TFT": tft_variance,
    "NHits": nhits_variance,
    "DLinear": dlinear_variance,
    "NLinear": nlinear_variance,
    "TSMixer": tsmixer_variance
}

# Plotting the variances as a bar chart
fig, ax = plt.subplots(figsize=(10, 4))

custom_colors = ["#c0c0c0", "#1f77b4", "#c0c0c0", "#c0c0c0", "#c0c0c0", "#c0c0c0", "#c0c0c0", "#800080", "#D3D3D3", "#00FFFF"]

# Create bar chart
bars = ax.bar(variances.keys(), variances.values(), color=custom_colors[:7], alpha=0.8)

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

# Add title and labels
ax.set_ylabel("Var Time Diff (hour)", fontsize=18)

# Customize ticks for readability
ax.tick_params(axis='x', labelsize=18, rotation=45)
ax.tick_params(axis='y', labelsize=18)

# Add variance values on top of bars
#for i, v in enumerate(variances.values()):
#    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=14)

# Show the plot
plt.tight_layout()
plt.savefig("fig_7c.png")
#plt.show()

