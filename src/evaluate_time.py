import matplotlib.pyplot as plt
import pandas as pd

# Color mapping
custom_colors = {
    'LSTM': "#1f77b4",
    'N-BEATS': "#1f77b4",
    'TFT': "#1f77b4",
    'NHits': "#1f77b4",
    'DLinear': "#1f77b4",
    'NLinear': "#1f77b4",
    'TSMixer': "#1f77b4"
}

def plot_time_bar(data_dict, ylabel, filename):
    # Create DataFrame
    df = pd.DataFrame(data_dict)

    # Calculate absolute values and means
    means = df.abs().mean()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(
        means.index,
        means.values,
        color=[custom_colors.get(label, "#1f77b4") for label in means.index],
        alpha=0.8
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=18
        )

    # Clean up spines and add grid
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.7)
    ax.xaxis.grid(True, linestyle='-', alpha=0.7)

    # Labels and ticks
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='x', labelsize=18, rotation=45)
    ax.tick_params(axis='y', labelsize=18)

    # Save
    plt.tight_layout()
    plt.savefig(filename)
    #plt.show()

# === Plot Training Time (hr) ===
training_time_data = {
    'LSTM': [2.07],
    'N-BEATS': [20.29],
    'TFT': [24.79],
    'NHits': [2.78],
    'DLinear': [1.01],
    'NLinear': [1.03],
    'TSMixer': [7.60]
}
plot_time_bar(training_time_data, ylabel="Model Training Time (hr)", filename="fig_9a.png")

# === Plot Execution Time (s) ===
execution_time_data = {
    'LSTM': [3.40],
    'N-BEATS': [5.57],
    'TFT': [3.15],
    'NHits': [3.76],
    'DLinear': [2.48],
    'NLinear': [2.29],
    'TSMixer': [2.41]
}
plot_time_bar(execution_time_data, ylabel="Execution Time (s)", filename="fig_9b.png")

