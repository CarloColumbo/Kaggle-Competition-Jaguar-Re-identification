import torch
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models import BaseModel


@torch.no_grad()
def extract_embeddings(
    model: BaseModel,
    image_paths: list[str],
    batch_size: int,
    device: str,
    desc: str = "Extracting embeddings"
):
    """
    Extract embeddings for a list of image paths.
    Args:
        model (BaseModel): The model used to extract embeddings.
        image_paths (list[str]): List of image file paths.
        batch_size (int): Number of images to process in a batch.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        desc (str): Description for the progress bar.
    Returns:
        np.ndarray: Array of extracted embeddings.
    """
    model.eval()
    embeddings = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc=desc):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess batch
        batch_tensors = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = model.preprocess(img)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Use zero tensor as fallback
                batch_tensors.append(torch.zeros(3, model.get_input_size(), model.get_input_size()))
        
        # Stack and move to device
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Get embeddings
        batch_emb = model(batch_tensor).cpu().numpy()
        embeddings.append(batch_emb)
    
    return np.vstack(embeddings)
    

def _load_cached_embeddings(
    cache_path: Path,
    expected_filenames: list[str]
) -> Optional[np.ndarray]:
    """
    Load cached embeddings from a .npz file.
    """
    z = np.load(cache_path, allow_pickle=True)
    cached_embeddings = z["embeddings"]
    cached_filenames = z["filenames"].tolist() if isinstance(z["filenames"], np.ndarray) else list(z["filenames"])

    if len(cached_filenames) != len(expected_filenames):
        return None

    if set(cached_filenames) != set(expected_filenames):
        return None

    if cached_filenames == expected_filenames:
        return cached_embeddings

    idx = {fn: i for i, fn in enumerate(cached_filenames)}
    return np.stack([cached_embeddings[idx[fn]] for fn in expected_filenames], axis=0)
    
    
def get_embeddings(
    model: BaseModel,
    image_paths: list[str],
    device: str,
    batch_size: int,
    cache_path: Path = None,
    desc: str = "Extracting embeddings"
) -> np.ndarray:
    """
    Get embeddings for a list of image paths, using cached embeddings if available.
    Args:
        model (BaseModel): The model used to extract embeddings.
        image_paths (list[str]): List of image file paths.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        batch_size (int): Number of images to process in a batch.
        cache_path (Path, optional): Path to cache file for embeddings. If None, caching is disabled.
        desc (str): Description for the progress bar.
    Returns:
        np.ndarray: Array of extracted embeddings.
    """
    embeddings = None
    if cache_path and cache_path.exists():
        embeddings = _load_cached_embeddings(cache_path, image_paths)
        if embeddings is not None:
            print(f"Loaded cached embeddings from {cache_path}")
            print(f"Embeddings shape: {embeddings.shape}")

    if embeddings is None:
        print(f"Extracting embeddings for {len(image_paths)} images...")
        embeddings = extract_embeddings(
            model,
            image_paths,
            batch_size=batch_size,
            device=device,
            desc=desc
        )
        if cache_path:
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                filenames=np.array(image_paths, dtype=object)
            )
            print(f"Saved embeddings to cache at {cache_path}")

    return embeddings
