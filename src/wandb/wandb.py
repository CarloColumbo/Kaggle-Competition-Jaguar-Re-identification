import wandb
from pathlib import Path
import pandas as pd


def log_train_val_distribution(train_data: pd.DataFrame, val_data: pd.DataFrame, train_counts: pd.Series, val_counts: pd.Series):
    """
    Log the training and validation identity distribution to W&B.
    Args:
        train_data (pd.DataFrame): Training dataset DataFrame.
        val_data (pd.DataFrame): Validation dataset DataFrame.
        train_counts (pd.Series): Series with counts of images per identity in the training set.
        val_counts (pd.Series): Series with counts of images per identity in the validation set.
    """
    num_classes = len(train_counts)

    distribution_df = pd.DataFrame({
        'identity': train_counts.index,
        'train_count': train_counts.values,
        'val_count': val_counts.values,
        'total_count': train_counts.values + val_counts.values,
        'train_ratio': train_counts.values / (train_counts.values + val_counts.values)
    })
    
    # Log table and summary stats to W&B
    wandb.log({
        "identity_distribution_table": wandb.Table(dataframe=distribution_df),
        "num_identities": num_classes,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_samples_per_identity": wandb.Histogram(train_counts.values),
        "val_samples_per_identity": wandb.Histogram(val_counts.values),
    })

    print(f"\nLogged identity distributions to W&B")
    print(f"  Train samples per identity: {train_counts.min()} - {train_counts.max()} (mean: {train_counts.mean():.1f})")
    print(f"  Val samples per identity: {val_counts.min()} - {val_counts.max()} (mean: {val_counts.mean():.1f})")


def add_model_artifact(model_path: Path):
    """
    Save the model file as a W&B artifact.
    Args:
        model_path (Path): Path to the model file.
    """
    model_artifact = wandb.Artifact(
        name="arcface-model",
        type="model",
        description="ArcFace fine-tuned MegaDescriptor model for jaguar re-identification"
    )
    model_artifact.add_file(str(model_path))
    wandb.log_artifact(model_artifact)

    print("Model artifact saved to W&B")


def add_submission_artifact(submission_path: Path):
    """
    Save the submission file as a W&B artifact.
    Args:
        submission_path (Path): Path to the submission file.
    """
    submission_artifact = wandb.Artifact(
        name="submission",
        type="submission",
        description="Competition submission file"
    )
    submission_artifact.add_file(str(submission_path))
    wandb.log_artifact(submission_artifact)

    print("Submission artifact saved to W&B")
