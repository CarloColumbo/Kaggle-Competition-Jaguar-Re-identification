import torch
from torch import nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    """
    Proxy Anchor Loss implementation.

    This loss uses class proxies to compute the loss for each sample in the batch.

    The loss is computed as:
        L = log(1 + sum(exp(-alpha * (sim - margin)))) + log(1 + sum(exp(alpha * (sim + margin))))

    where:
        - sim is the cosine similarity between features and proxies
        - alpha is a scaling factor
        - margin is a margin applied to the similarity scores

    Args:
        num_classes (int): Number of classes.
        embedding_dim (int): Dimension of the embedding space.
        margin (float): Margin for similarity scores. Default is 0.1.
        alpha (float): Scaling factor. Default is 32.
    """
    def __init__(self, num_classes, embedding_dim, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin
        self.alpha = alpha

    def forward(self, features, labels):
        features = F.normalize(features)
        proxies = F.normalize(self.proxies)

        sim = F.linear(features, proxies)  # cosine similarity

        pos_mask = F.one_hot(labels, num_classes=sim.size(1)).bool()
        neg_mask = ~pos_mask

        pos_exp = torch.exp(-self.alpha * (sim - self.margin)) * pos_mask
        neg_exp = torch.exp(self.alpha * (sim + self.margin)) * neg_mask

        pos_term = torch.log(1 + pos_exp.sum(dim=0)).sum()
        neg_term = torch.log(1 + neg_exp.sum(dim=0)).sum()

        loss = (pos_term + neg_term) / features.size(0)
        return loss
