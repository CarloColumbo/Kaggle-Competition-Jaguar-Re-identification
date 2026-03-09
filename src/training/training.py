import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb


def compute_validation_map(model, val_loader, device):
    """
    Compute identity-balanced mean Average Precision on validation set.
    
    This simulates the competition metric:
    1. For each query, rank all other images by cosine similarity
    2. Compute Average Precision based on where true matches appear
    3. Average APs within each identity, then average across identities
    """
    model.eval()
    
    val_embeddings = []
    val_labels = []
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            embeddings = model(data)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            val_embeddings.append(embeddings.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_embeddings = np.concatenate(val_embeddings)
    val_labels = np.concatenate(val_labels)

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(val_embeddings)
    np.fill_diagonal(sim_matrix, -1)  # Exclude self-similarity

    # Compute AP for each query
    query_aps = {}
    
    for query_idx in range(len(val_labels)):
        query_label = val_labels[query_idx]
        
        # Get similarities to all gallery images (excluding self)
        similarities = sim_matrix[query_idx]
        
        # True labels for gallery
        gallery_labels = val_labels.copy()
        is_match = (gallery_labels == query_label).astype(int)
        is_match[query_idx] = 0  # Exclude self
        
        # Sort by similarity descending
        sorted_indices = np.argsort(-similarities)
        sorted_matches = is_match[sorted_indices]
        
        # Compute Average Precision
        n_positives = sorted_matches.sum()
        if n_positives == 0:
            continue
        
        cumsum = np.cumsum(sorted_matches)
        precision_at_k = cumsum / np.arange(1, len(sorted_matches) + 1)
        ap = np.sum(precision_at_k * sorted_matches) / n_positives
        
        query_aps[query_idx] = (query_label, ap)
    
    # Group by identity and compute identity-balanced mAP
    identity_aps = {}
    for query_idx, (label, ap) in query_aps.items():
        if label not in identity_aps:
            identity_aps[label] = []
        identity_aps[label].append(ap)
    
    # Average within identity, then across identities
    identity_mean_aps = [np.mean(aps) for aps in identity_aps.values()]
    balanced_map = np.mean(identity_mean_aps)
    
    return balanced_map


def train_epoch(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device
    ):
    """
    Train for one epoch.
    
    Args:
        model (nn.Model): the neural network model to train
        loader (DataLoader): training data loader
        criterion (nn.Module): loss function
        optimizer (Optimizer): optimizer for updating model parameters
        scheduler (LRScheduler): learning rate scheduler to update learning rate
        device (torch.device): device to run training on
    
    Returns:
        avg_loss (float): average training loss for the epoch
    """
    model.train()
    total_loss = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        embeddings = model(data)
        loss = criterion(embeddings, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # If OneCycleLR, step per batch
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    return avg_loss


def validate_epoch(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
    ):
    """
    Validate for one epoch.
    
    Args:
        model (nn.Model): the neural network model to validate
        loader (DataLoader): validation data loader
        criterion (nn.Module): loss function
        device (torch.device): device to run validation on
        
    Returns:
        avg_loss (float): average validation loss for the epoch
    """
    model.eval()
    total_loss = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)

            embeddings = model(data)
            loss = criterion(embeddings, labels)

            total_loss += loss.item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    return avg_loss
    

def train_loop(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        name: str,
        checkpoint_path: str,
        num_epochs: int,
        patience: int,
        classes: list,
        silent: bool=False
    ):
    """
    Train the model with early stopping and checkpointing.
    Uses wandb logging.
    
    Args:
        model (nn.Module): the neural network model to train
        train_loader (DataLoader): training data loader
        val_loader (DataLoader): validation data loader
        criterion (nn.Module): loss function
        optimizer (Optimizer): optimizer for updating model parameters
        scheduler (LRScheduler): learning rate scheduler
        device (torch.device): device to run training
        name (str): name for logging and checkpointing
        checkpoint_path (str): path to save the best model checkpoint
        num_epochs (int): maximum number of epochs to train
        patience (int): number of epochs to wait for improvement before early stopping
        classes (list): list of class labels for the dataset
        silent (bool): if True, suppress training logs (except final summary)
    """

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_map': [], 'lr': []
    }

    best_val_map = 0.0
    patience_counter = 0
    best_epoch = 0

    if not silent:
        print(f"Starting training for {num_epochs} epochs...")
        print("=" * 70)

    for epoch in range(num_epochs):
        if not silent:
            print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Compute validation mAP
        val_map = compute_validation_map(
            model, 
            val_loader,
            device,
        )
        
        # Update scheduler
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map)
        history['lr'].append(current_lr)
        
        # Log to W&B
        wandb.log({
            'model': name,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_map': val_map,
            'learning_rate': current_lr,
        })
        
        # Print summary
        if not silent:
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val mAP:    {val_map:.4f} | LR: {current_lr:.2e}")

        # Checkpoint best model
        if val_map > best_val_map:
            best_val_map = val_map
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_map': val_map,
                'num_epochs': num_epochs,
                'patience': patience,
                'label_encoder_classes': classes,
                'num_classes': len(classes),
                'name': name
            }, checkpoint_path)

            if not silent:
                print(f"  [New best model saved]")
        else:
            patience_counter += 1
            if not silent:
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            if not silent:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    if not silent:
        print("\n" + "=" * 70)
        print(f"Training complete!")
        print(f"Best epoch: {best_epoch}, Val mAP: {best_val_map:.4f}")

    return history, best_val_map, best_epoch
