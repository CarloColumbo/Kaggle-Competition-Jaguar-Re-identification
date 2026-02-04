import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def get_train_df(path: Path, info: bool = True) -> pd.DataFrame:
    """
    Loads the training dataframe from a CSV file and optionally prints dataset information.
    Args:
        path (Path): Path to the CSV file containing the training data.
        info (bool): Whether to print dataset information. Default is True.
    Returns:
        pd.DataFrame: The loaded training dataframe.
    """
    train_df = pd.read_csv(path)

    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['ground_truth'])

    if info:
        print(f"Training dataset:")
        print(f"  Total images: {len(train_df)}")
        print(f"  Unique identities: {train_df['ground_truth'].nunique()}")
        print(f"\nSample rows:")
        print(train_df.head())

    return train_df


def get_identity_counts(train_df: pd.DataFrame, info: bool = True) -> pd.Series:
    """
    Computes the counts of images per jaguar identity in the training dataframe.
    Args:
        train_df (pd.DataFrame): The training dataframe containing 'ground_truth' column.
        info (bool): Whether to print identity distribution information. Default is True.
    Returns:
        pd.Series: Series with counts of images per jaguar identity.
    """
    identity_counts = train_df['ground_truth'].value_counts()
    
    if info:
        print(f"\nIdentity distribution:")
        print(f"  Min images per identity: {identity_counts.min()} ({identity_counts.idxmin()})")
        print(f"  Max images per identity: {identity_counts.max()} ({identity_counts.idxmax()})")
        print(f"  Mean images per identity: {identity_counts.mean():.1f}")
        
    # Identify identities that may need careful handling (few samples)
    min_samples_for_split = 2  # Need at least 2 to split
    low_sample_identities = identity_counts[identity_counts < min_samples_for_split]

    if len(low_sample_identities) > 0:   
        print(f"\nWarning: {len(low_sample_identities)} identities have fewer than {min_samples_for_split} images")
        
    return identity_counts