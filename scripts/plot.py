import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import MaxNLocator

# Function to create the plot
def plot_fitness_log(file_path):
    # Load the CSV data
    data = pd.read_csv(file_path)

    # Customize plot style
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.size': 12,          # Set font size
        'font.family': 'sans-serif',
        'grid.color': 'gray',     # Grid color
        'grid.linestyle': '--',   # Grid style
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'legend.fontsize': 10
    })

    # Create the plot with two subplots, one on top of the other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot RMSE on the top axis with a logarithmic scale
    ax1.plot(data['RMSE'], marker='o', linestyle='-', color='cyan', label='RMSE')
    ax1.set_yscale('log')  # Set the y-axis to a log scale
    ax1.set_ylabel('RMSE (Log Scale)', color='cyan', fontsize=12)
    ax1.tick_params(axis='y', colors='cyan')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    # Plot Penalty on the bottom axis
    ax2.plot(data['Penalty'], marker='x', linestyle='-', color='orange', label='Term Count')
    ax2.set_xlabel('Iteration', color='white', fontsize=12)
    ax2.set_ylabel('Term Count', color='orange', fontsize=12)
    ax2.tick_params(axis='y', colors='orange')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # Force y-axis to show integers only
    ax2.grid(True)
    ax2.legend(loc='upper right')

    # Title for the whole figure
    fig.suptitle('RMSE and Penalty Over Iterations', color='white', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to fit the title properly

    plt.show()

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot RMSE and Penalty from a fitness log CSV.')
    parser.add_argument('file_path', type=str, help='Path to the fitness log CSV file')
    
    args = parser.parse_args()
    plot_fitness_log(args.file_path)
