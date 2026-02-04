import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb


def visualize_train_val_distribution(train_counts: pd.Series, val_counts: pd.Series, log: bool = True) -> plt.Figure:
    """
    Visualizes the distribution of images per identity in the training and validation sets.
    Args:
        train_counts (pd.Series): Series with counts of images per identity in the training set.
        val_counts (pd.Series): Series with counts of images per identity in the validation set.
        log (bool): Whether to log the figure to W&B. Default is True.
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
    
    if log:
        wandb.log({"train_val_distribution": wandb.Image(fig)})
    
    return fig
