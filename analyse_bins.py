import matplotlib.pyplot as plt
import numpy as np
from ArcBallDataset import ArcBallDatasetDiscrete, get_bin_configs

def plot_dataset_histograms(dataset, configs):
    states = dataset.s_idx.numpy()
    deltas = dataset.y_idx.numpy()
    
    state_edges = configs['state_edges']
    delta_edges = configs['delta_edges']

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle('Physical Value Distributions (Bin Centers)', fontsize=18, fontweight='bold')
    
    labels = ["Cart Position", "Cart Velocity", "Ball Position", "Ball Velocity"]

    for i in range(4):
        # Calculate centers: (left_edge + right_edge) / 2
        s_centers = (state_edges[i][:-1] + state_edges[i][1:]) / 2.0
        d_centers = (delta_edges[i][:-1] + delta_edges[i][1:]) / 2.0
        
        # Map indices to centers
        s_vals = s_centers[states[:, i]]
        d_vals = d_centers[deltas[:, i]]
        
        # --- Left Column: States ---
        axes[i, 0].hist(s_vals, bins=state_edges[i], color='skyblue', edgecolor='black', alpha=0.7)
        axes[i, 0].set_title(f'State: {labels[i]}', fontsize=12)
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].grid(axis='y', linestyle='--', alpha=0.3)
        
        # --- Right Column: Deltas ---
        axes[i, 1].hist(d_vals, bins=delta_edges[i], color='salmon', edgecolor='black', alpha=0.7)
        axes[i, 1].set_title(f'Delta: {labels[i]}', fontsize=12)
        axes[i, 1].grid(axis='y', linestyle='--', alpha=0.3)

    # Set bottom labels
    for ax in axes[3, :]:
        ax.set_xlabel('Physical Value (Units)')

    # --- ADJUST SPACING HERE ---
    # hspace: height padding between subplots (0.4 is a good starting point)
    # wspace: width padding between columns
    plt.subplots_adjust(hspace=0.4, wspace=0.2, top=0.92, bottom=0.08)
    
    plt.show()

if __name__ == "__main__":
    path = "/home/snath/data/dataset/arcball_discrete_600k.h5"
    configs = get_bin_configs(path, target_bins=200)
    ds = ArcBallDatasetDiscrete(path, bin_configs=configs)
    
    plot_dataset_histograms(ds, configs)