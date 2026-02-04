import matplotlib.pyplot as plt
import pandas as pd
import wandb


def visualize_identity_distribution(identity_counts: pd.Series, log: bool = True) -> plt.Figure:
    """
    Visualizes the distribution of images per jaguar identity in the training dataset.
    Logs the visualization to Weights & Biases (W&B).

    Args:
        identity_counts (pd.Series): Series with counts of images per jaguar identity.
        log (bool): Whether to log the figure to W&B. Default is True.
    Returns:
        plt.Figure: The generated matplotlib figure.
    """

    fig, ax = plt.subplots(figsize=(14, 5))
    identity_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel('Jaguar Identity')
    ax.set_ylabel('Number of Images')
    ax.set_title('Training Data: Images per Jaguar Identity')
    ax.axhline(y=identity_counts.mean(), color='red', linestyle='--', label=f'Mean: {identity_counts.mean():.1f}')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.show()
    
    if log:
        wandb.log({"identity_distribution_full": wandb.Image(fig)})

    return fig
