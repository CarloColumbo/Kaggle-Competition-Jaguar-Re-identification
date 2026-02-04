import matplotlib.pyplot as plt
import pandas as pd


def visualize_identity_distribution(identity_counts: pd.Series) -> plt.Figure:
    """
    Visualizes the distribution of images per jaguar identity in the training dataset.
    Logs the visualization to Weights & Biases (W&B).

    Args:
        identity_counts (pd.Series): Series with counts of images per jaguar identity.
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

    return fig
