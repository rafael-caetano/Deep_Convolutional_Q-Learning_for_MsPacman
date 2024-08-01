import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_tiled_performance_comparison(original_scores_list, dueling_scores_list, window_size=100):
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    fig.suptitle('Performance Comparison - All Runs', fontsize=16, y=0.95)

    for i, (original_scores, dueling_scores) in enumerate(zip(original_scores_list, dueling_scores_list)):
        row = i // 2
        col = i % 2
        
        # Calculate rolling averages
        original_avg = np.convolve(original_scores, np.ones(window_size)/window_size, mode='valid')
        dueling_avg = np.convolve(dueling_scores, np.ones(window_size)/window_size, mode='valid')

        # Plot
        axs[row, col].plot(original_avg, label='Original DQN')
        axs[row, col].plot(dueling_avg, label='Dueling DQN')
        axs[row, col].set_xlabel('Episode')
        axs[row, col].set_ylabel(f'Average Score (over {window_size} episodes)')
        axs[row, col].set_title(f'Run {i+1}')
        axs[row, col].legend()
        axs[row, col].grid(True)

    # Remove the unused subplot
    fig.delaxes(axs[2, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the title
    
    # Save the plot
    filename = 'outputs/tiled_performance_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Tiled performance comparison plot saved as {filename}")
    
    # Display the plot
    plt.show()

def load_scores(model_name, num_runs):
    scores_list = []
    for i in range(1, num_runs + 1):
        with open(f'{model_name}_scores_run_{i}.pkl', 'rb') as f:
            scores = pickle.load(f)
        scores_list.append(scores)
    return scores_list

# Ensure the outputs directory exists
os.makedirs('outputs', exist_ok=True)