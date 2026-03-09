from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def extract_embeddings(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        desc: str = "Extracting embeddings"
    ):
    """
    Extract embeddings for images.
    
    Args:
        model: The model to use for embedding extraction.
        data_loader: DataLoader providing batches of images.
        device: Device to perform computations on.
        desc: Description for the progress bar.
        
    Returns:
        A numpy array of shape (num_images, embedding_dim) containing the extracted embeddings.
    """
    model.eval()
    embeddings = []

    for batch in tqdm(data_loader, desc=desc):
        batch = batch.to(device)
        with torch.no_grad():
            emb = model(batch)
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
    

def _load_cached_embeddings(
        cache_path: Path,
        expected_filenames: list[str]
    ):
    """
    Load cached embeddings if they match the expected filenames.
    
    Args:
        cache_path: Path to the cached embeddings file.
        expected_filenames: List of filenames that the cached embeddings should correspond to.
        
    Returns:
        Cached embeddings if they match the expected filenames, otherwise None.
    """
    z = np.load(cache_path, allow_pickle=True)
    cached_embeddings = z["embeddings"]
    cached_filenames = z["filenames"].tolist() if isinstance(z["filenames"], np.ndarray) else list(z["filenames"])

    if len(cached_filenames) != len(expected_filenames):
        return None

    if set(cached_filenames) != set(expected_filenames):
        return None

    idx = {fn: i for i, fn in enumerate(cached_filenames)}
    return np.stack([cached_embeddings[idx[fn]] for fn in expected_filenames], axis=0)
    
    
def get_embeddings(
        model: nn.Module,
        cache_path: Path,
        data_loader: torch.utils.data.DataLoader,
        filenames: list[str],
        device: torch.device
    ):
    """
    Get embeddings for a list of filenames, using cached embeddings if available.
    
    Args:
        model: The model to use for embedding extraction.
        cache_path: Path to the cached embeddings file.
        data_loader: DataLoader providing batches of images.
        filenames: List of filenames to get embeddings for.
        device: Device to perform computations on.
        
    Returns:
        A numpy array of shape (num_images, embedding_dim) containing the embeddings for the specified
    """
    embeddings = None
    if cache_path.exists():
        embeddings = _load_cached_embeddings(cache_path, filenames)
        if embeddings is not None:
            print(f"Loaded cached embeddings from {cache_path}")
            print(f"Embeddings shape: {embeddings.shape}")
            
    if embeddings is None:
        print(f"Extracting embeddings for {len(data_loader)} images...")
        embeddings = extract_embeddings(
            model,
            data_loader,
            device=device,
        )
        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            filenames=np.array(filenames, dtype=object),
        )
        print(f"Saved embeddings cache to {cache_path}")
        print(f"Embeddings shape: {embeddings.shape}")
        
    return embeddings
