import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_split(
    train_df: pd.DataFrame,
    test_size: float,
    random_state: int = 42,
    info: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the training DataFrame into training and validation sets.
    Args:
        train_df (pd.DataFrame): DataFrame containing training data with 'ground_truth' column.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        info (bool, optional): If True, prints information about the split. Defaults to True.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
    """
    train_data, val_data = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df["ground_truth"],
    )
    
    if info:
        print(f"Dataset split:")
        print(f"  Training:   {len(train_data)} images ({100*(1-test_size):.0f}%)")
        print(f"  Validation: {len(val_data)} images ({100*test_size:.0f}%)")

        # Verify all identities are in both sets
        train_identities = set(train_data['ground_truth'].unique())
        val_identities = set(val_data['ground_truth'].unique())

        print(f"\nIdentity coverage:")
        print(f"  Identities in training:   {len(train_identities)}")
        print(f"  Identities in validation: {len(val_identities)}")
        print(f"  Overlap: {len(train_identities & val_identities)}")

    return train_data, val_data
