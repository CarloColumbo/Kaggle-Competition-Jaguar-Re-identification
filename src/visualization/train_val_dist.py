import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_train_val_distribution(train_counts: pd.Series, val_counts: pd.Series):
    """
    Visualizes the distribution of images per identity in the training and validation sets.
    Args:
        train_counts (pd.Series): Series with counts of images per identity in the training set.
        val_counts (pd.Series): Series with counts of images per identity in the validation set.
    Returns:
        fig: Matplotlib figure object containing the visualization.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.35
    x = np.arange(len(train_counts))
    ax.bar(x - width/2, train_counts.values, width, label='Train', color='steelblue')
    ax.bar(x + width/2, val_counts.values, width, label='Validation', color='coral')
    ax.set_xlabel('Jaguar Identity')
    ax.set_ylabel('Number of Images')
    ax.set_title('Train vs Validation: Images per Identity')
    ax.set_xticks(x)
    ax.set_xticklabels(train_counts.index, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    plt.show()
    
    return fig
