import pandas as pd
import matplotlib.pyplot as plt

def visualize_results(results_file):
    # Read the results
    df = pd.read_csv(results_file)
    
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot MABO
    ax1.plot(df['num_rects'], df['mabo_edge'], '-o', label='EdgeBoxes', alpha=0.7)
    ax1.plot(df['num_rects'], df['mabo_ss_fast'], '-o', label='SS Fast', alpha=0.7)
    ax1.plot(df['num_rects'], df['mabo_ss_qual'], '-o', label='SS Quality', alpha=0.7)
    ax1.set_xlabel('Number of Proposals')
    ax1.set_ylabel('MABO')
    ax1.set_title('Mean Average Best Overlap by Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Recalls
    k_values = [0.5, 0.7, 0.9]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Default matplotlib colors
    
    for i, k in enumerate(k_values):
        ax2.plot(df['num_rects'], df[f'recall_edge_k{k}'], '-o', 
                color=colors[i], label=f'EdgeBoxes (k={k})', alpha=0.7)
        ax2.plot(df['num_rects'], df[f'recall_ss_fast_k{k}'], '--o', 
                color=colors[i], label=f'SS Fast (k={k})', alpha=0.7)
        ax2.plot(df['num_rects'], df[f'recall_ss_qual_k{k}'], ':o', 
                color=colors[i], label=f'SS Quality (k={k})', alpha=0.7)
    
    ax2.set_xlabel('Number of Proposals')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall by Method and IoU Threshold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('region_proposal_results.png', bbox_inches='tight', dpi=300)
    plt.close()

# Alternative visualization with separate recall plots
def visualize_results_detailed(results_file):
    # Read the results
    df = pd.read_csv(results_file)
    
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot MABO
    ax1.plot(df['num_rects'], df['mabo_edge'], '-o', label='EdgeBoxes', alpha=0.7)
    ax1.plot(df['num_rects'], df['mabo_ss_fast'], '-o', label='SS Fast', alpha=0.7)
    ax1.plot(df['num_rects'], df['mabo_ss_qual'], '-o', label='SS Quality', alpha=0.7)
    ax1.set_xlabel('Number of Proposals')
    ax1.set_ylabel('MABO')
    ax1.set_title('Mean Average Best Overlap by Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Recalls for each k value separately
    k_values = [0.5, 0.7, 0.9]
    axes = [ax2, ax3, ax4]
    
    for ax, k in zip(axes, k_values):
        ax.plot(df['num_rects'], df[f'recall_edge_k{k}'], '-o', 
                label='EdgeBoxes', alpha=0.7)
        ax.plot(df['num_rects'], df[f'recall_ss_fast_k{k}'], '-o', 
                label='SS Fast', alpha=0.7)
        ax.plot(df['num_rects'], df[f'recall_ss_qual_k{k}'], '-o', 
                label='SS Quality', alpha=0.7)
        ax.set_xlabel('Number of Proposals')
        ax.set_ylabel('Recall')
        ax.set_title(f'Recall by Method (IoU = {k})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('region_proposal_results_detailed.png', bbox_inches='tight', dpi=300)
    plt.close()

# Example usage:
if __name__ == "__main__":
    # You can call either or both visualization functions
    visualize_results('results.csv')
    visualize_results_detailed('results.csv')