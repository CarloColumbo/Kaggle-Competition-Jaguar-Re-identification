import torch
from torch import nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss implementation.

    This loss selects the hardest positive and hardest negative for each anchor in the batch to compute the triplet loss.

    The loss is computed as:
        L = max(0, d(a, p) - d(a, n) + margin)

    where:
        - d(a, p) is the distance between the anchor and the hardest positive
        - d(a, n) is the distance between the anchor and the hardest negative
        - margin is a hyperparameter that enforces a minimum separation between positive and negative pairs

    Args:
        margin (float): Margin for triplet loss. Default is 0.3.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, embedding_dim) - feature embeddings of the batch
            labels: (batch_size,) - class labels for each embedding

        Returns:
            loss: scalar - Batch Hard Triplet Loss
        """
        features = F.normalize(features)
        dist = torch.cdist(features, features, p=2)

        labels = labels.unsqueeze(1)
        mask_pos = labels.eq(labels.T)
        mask_neg = ~mask_pos

        hardest_pos = (dist * mask_pos.float()).max(dim=1)[0]

        dist_neg = dist.clone()
        dist_neg[~mask_neg] = float('inf')
        hardest_neg = dist_neg.min(dim=1)[0]

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()
