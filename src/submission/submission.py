import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


def get_test_pairs(data_path: Path, info: bool = True) -> pd.DataFrame:
    """
    Load the sample submission file and return the test pairs as a DataFrame.
    Args:
        file_path (Path): Path to the sample submission CSV file.
        info (bool): Whether to print information about the loaded DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing the test pairs.
    """
    test_pairs_df = pd.read_csv(data_path / "test.csv")

    if info:
        print(f"Test pairs to score: {len(test_pairs_df)}")
        print(f"Columns: {list(test_pairs_df.columns)}")
        print(f"\nSample rows:")
        print(test_pairs_df.head())

    return test_pairs_df


def get_test_images(test_pairs_df: pd.DataFrame) -> set:
    """
    Extract unique test image filenames from the test pairs DataFrame.
    Args:
        test_pairs_df (pd.DataFrame): DataFrame containing the test pairs.
    Returns:
        set: Set of unique test image filenames.
    """
    test_images = set(test_pairs_df['query_image'].unique()) | set(test_pairs_df['gallery_image'].unique())
    test_images = sorted(list(test_images))
    return test_images


def compute_pairwise_similarities(test_pairs_df: pd.DataFrame, img_to_embedding: dict, info: bool = True) -> np.ndarray:
    """
    Compute pairwise similarities for the given test pairs.
    Args:
        test_pairs_df (pd.DataFrame): DataFrame containing the test pairs.
        img_to_embedding (dict): Dictionary mapping image filenames to their embeddings.
        info (bool): Whether to print similarity statistics.
    Returns:
        np.ndarray: Array of similarity scores.
    """
    # Compute similarity for each pair
    if info:
        print("Computing pairwise similarities...")

    similarities = []
    for _, row in tqdm(test_pairs_df.iterrows(), total=len(test_pairs_df), desc="Computing similarities"):
        query_emb = img_to_embedding[row['query_image']]
        gallery_emb = img_to_embedding[row['gallery_image']]

        # Cosine similarity (embeddings are already normalized)
        sim = np.dot(query_emb, gallery_emb)
        similarities.append(sim)
    
    # Clip to [0, 1] range
    similarities = np.array(similarities)
    similarities = np.clip(similarities, 0.0, 1.0)

    if info:
        print(f"\nSimilarity statistics:")
        print(f"  Min: {similarities.min():.4f}")
        print(f"  Max: {similarities.max():.4f}")
        print(f"  Mean: {similarities.mean():.4f}")
        print(f"  Std: {similarities.std():.4f}")

    return similarities


def create_submission_dataframe(
    test_pairs_df: pd.DataFrame,
    similarities: np.ndarray,
    data_path: Path,
    info: bool = True,
    verify: bool = True
) -> pd.DataFrame:
    """
    Create a submission CSV file with the computed similarities.
    Args:
        test_pairs_df (pd.DataFrame): DataFrame containing the test pairs.
        similarities (np.ndarray): Array of similarity scores.
        data_path (Path): Path to the data directory containing the sample submission.
        info (bool): Whether to print submission information.
        verify (bool): Whether to verify the submission format against the sample submission.
    Returns:
        pd.DataFrame: DataFrame containing the submission data.
    """
    submission_df = pd.DataFrame({
        'row_id': test_pairs_df['row_id'],
        'similarity': similarities
    })
    
    if info:
        print("Submission DataFrame:")
        print(submission_df.head(10))

    # Verify format matches sample submission
    if verify:
        sample_submission = pd.read_csv(data_path / "sample_submission.csv")
        print(f"\nFormat check:")
        print(f"  Expected columns: {list(sample_submission.columns)}")
        print(f"  Our columns: {list(submission_df.columns)}")
        print(f"  Expected rows: {len(sample_submission)}")
        print(f"  Our rows: {len(submission_df)}")
        
    return submission_df
