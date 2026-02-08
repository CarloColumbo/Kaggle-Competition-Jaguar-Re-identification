import torch
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from src.models import BaseModel


@torch.no_grad()
def extract_embeddings(
    model: BaseModel,
    preprocess: transforms.Compose,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    desc: str = "Extracting embeddings"
):
    """
    Extract embeddings for a list of image paths.
    Args:
        model (BaseModel): The model used to extract embeddings.
        preprocess (transforms.Compose): Preprocessing transformations to apply to images.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        desc (str): Description for the progress bar.
    Returns:
        np.ndarray: Array of extracted embeddings.
    """
    model.eval()
    embeddings = []
    
    # for i in tqdm(range(0, len(datal), batch_size), desc=desc):
    for batch, _ in tqdm(dataloader, desc=desc):
        if preprocess:
            batch = preprocess(batch)
        else:
            batch = torch.stack([transforms.ToTensor()(img) for img in batch])

        batch_tensor = batch.to(device)
        
        # Get embeddings
        batch_emb = model.get_embeddings(batch_tensor).cpu().numpy()
        embeddings.append(batch_emb)
    
    return np.vstack(embeddings)
    

def _load_cached_embeddings(
    cache_path: Path,
    dataloader: torch.utils.data.DataLoader
) -> Optional[np.ndarray]:
    """
    Load cached embeddings from a .npz file.
    """
    z = np.load(cache_path, allow_pickle=True)
    cached_embeddings = z["embeddings"]
    return cached_embeddings
    # cached_filenames = z["filenames"].tolist() if isinstance(z["filenames"], np.ndarray) else list(z["filenames"])
    
    # expected_filenames = [str(path) for path, _ in dataloader.dataset]

    # if len(cached_filenames) != len(expected_filenames):
    #     return None

    # if set(cached_filenames) != set(expected_filenames):
    #     return None

    # if cached_filenames == expected_filenames:
    #     return cached_embeddings

    # idx = {fn: i for i, fn in enumerate(cached_filenames)}
    # return np.stack([cached_embeddings[idx[fn]] for fn in expected_filenames], axis=0)
    
    
def get_embeddings(
    model: BaseModel,
    preprocess: transforms.Compose,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    cache_path: Path = None,
    desc: str = "Extracting embeddings"
) -> np.ndarray:
    """
    Get embeddings for a list of image paths, using cached embeddings if available.
    Args:
        model (BaseModel): The model used to extract embeddings.
        preprocess (transforms.Compose): Preprocessing transformations to apply to images.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        batch_size (int): Number of images to process in a batch.
        cache_path (Path, optional): Path to cache file for embeddings. If None, caching is disabled.
        desc (str): Description for the progress bar.
    Returns:
        np.ndarray: Array of extracted embeddings.
    """
    embeddings = None
    if cache_path and cache_path.exists():
        embeddings = _load_cached_embeddings(cache_path, dataloader)
        if embeddings is not None:
            print(f"Loaded cached embeddings from {cache_path}")
            print(f"Embeddings shape: {embeddings.shape}")

    if embeddings is None:
        print(f"Extracting embeddings for {len(dataloader.dataset)} images...")
        embeddings = extract_embeddings(
            model,
            preprocess,
            dataloader,
            device=device,
            desc=desc
        )
        if cache_path:
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                # filenames=np.array(image_paths, dtype=object)
            )
            print(f"Saved embeddings to cache at {cache_path}")

    return embeddings
