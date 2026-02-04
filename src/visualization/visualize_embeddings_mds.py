import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


def compute_geodesic_distances(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute geodesic (angular) distance matrix for normalized embeddings.
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, n_features) containing the embeddings.
    Returns:
        np.ndarray: Geodesic distance matrix of shape (n_samples, n_samples).
    """
    # Normalize embeddings to unit sphere
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    
    # Compute cosine similarity
    cos_sim = np.clip(normalized @ normalized.T, -1.0, 1.0)
    
    # Convert to geodesic distance (arc length)
    geodesic_dist = np.arccos(cos_sim)
    
    return geodesic_dist


def visualize_embeddings_mds(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    seed: int = 42,
    max_samples: int = 500,
) -> plt.Figure:
    """
    Visualize embeddings using MDS with geodesic distances.
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, n_features) containing the embeddings.
        labels (np.ndarray): Array of shape (n_samples,) containing the identity labels.
        title (str): Title for the plot.
        max_samples (int, optional): Maximum number of samples to visualize. Defaults to 500.
    Returns:
        plt.Figure: Matplotlib figure containing the MDS plot.
    """
    # Subsample if too many points (MDS is O(n^3))
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Compute geodesic distance matrix
    dist_matrix = compute_geodesic_distances(embeddings)
    
    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed, normalized_stress='auto')
    coords_2d = mds.fit_transform(dist_matrix)
    
    # Create color mapping for identities
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords_2d[mask, 0], 
            coords_2d[mask, 1],
            c=[label_to_color[label]],
            label=label,
            alpha=0.7,
            s=30
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    
    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    return fig
