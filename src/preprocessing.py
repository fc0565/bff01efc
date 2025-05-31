
import zipfile
import pandas as pd
from darts import TimeSeries
from scipy.signal import savgol_filter

def process_zip_and_generate_series(zip_path, apply_smoothing=False,
                                     window_length=250, poly_order=3):
    # Load and sort
    dataframes = []
    with zipfile.ZipFile(zip_path, 'r') as zfile:
        csv_files = [f for f in zfile.namelist() if f.endswith('.csv')]
        for csv_file in csv_files:
            with zfile.open(csv_file) as file:
                df = pd.read_csv(file, index_col='timestamp')
                df.index = pd.to_datetime(df.index)
                dataframes.append(df)

    dfs_with_timestamp = [(df, df.index[0]) for df in dataframes]
    dfs_sorted = sorted(dfs_with_timestamp, key=lambda x: x[1])
    dfs1 = [df[0] for df in dfs_sorted]

    # Regularize + Filter short
    for i in range(len(dfs1)):
        dfs1[i] = dfs1[i].resample("100s").bfill()

    dfs1 = [df for df in dfs1 if len(df) >= 496]
    dfs1 = dfs1[:-1]  # remove last abnormal sample

    # Custom fixes (apply only if needed)
    fix_indices = {
        26: '2023-12-07 16:47:51',
        27: '2024-02-06 12:54:58',
        29: '2024-05-13 10:58:40',
        31: '2024-06-11 14:30:43',
        32: '2024-06-12 13:21:43'
    }

    for i, ts in fix_indices.items():
        if i < len(dfs1):
            dfs1[i] = dfs1[i].loc[ts:]

    # Convert to Darts TimeSeries
    series = []
    for df in dfs1:
        if apply_smoothing:
            smoothed = savgol_filter(df['value'], window_length=window_length, polyorder=poly_order)
            ts = TimeSeries.from_times_and_values(df.index, smoothed)
        else:
            ts = TimeSeries.from_dataframe(df, value_cols='value')
        series.append(ts)

    return series
