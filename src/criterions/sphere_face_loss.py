import math

import torch
from torch import nn
import torch.nn.functional as F


class SphereFaceLoss(nn.Module):
    """
    SphereFace / A-Softmax Loss implementation.

    This loss applies an angular margin to the logits for better class separation.

    The loss is computed as:
        L = -log(exp(s * cos(m * theta_y)) / (exp(s * cos(m * theta_y)) + sum(exp(s * cos(theta_j))))

    where:
        - theta_y is the angle between embedding and ground truth class center
        - m is the angular margin (integer >= 1)
        - s is the feature scale (default 64)

    Args:
        embedding_dim (int): Dimension of the embedding space.
        num_classes (int): Number of classes.
        margin (int): Angular margin. Default is 4.
        scale (float): Feature scale. Default is 64.0.
    """

    def __init__(self, embedding_dim, num_classes, margin=4, scale=64.0):
        super().__init__()

        assert isinstance(margin, int) and margin >= 1, \
            "SphereFace margin m must be an integer >= 1"

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.m = margin
        self.s = scale

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize features and weights
        x = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)

        # cos(theta)
        cosine = F.linear(x, W).clamp(-1.0, 1.0)

        # cos(m * theta) using Chebyshev polynomials
        cos_m_theta = self._cos_m_theta(cosine)

        # theta = arccos(cos(theta)) (only used to compute k, detached)
        theta = torch.acos(cosine.detach())

        # k = floor(m * theta / pi)
        k = (self.m * theta / math.pi).floor()

        # phi(theta) = (-1)^k * cos(mθ) - 2k
        phi = ((-1.0) ** k) * cos_m_theta - 2 * k

        # One-hot labels
        one_hot = F.one_hot(labels, self.num_classes).float()

        # Replace target logits
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s

        loss = F.cross_entropy(logits, labels)
        return loss

    def _cos_m_theta(self, cos_theta):
        """
        Compute cos(m * theta) via Chebyshev polynomials.
        """
        if self.m == 1:
            return cos_theta
        elif self.m == 2:
            return 2 * cos_theta**2 - 1
        elif self.m == 3:
            return 4 * cos_theta**3 - 3 * cos_theta
        elif self.m == 4:
            return 8 * cos_theta**4 - 8 * cos_theta**2 + 1
        elif self.m == 5:
            return 16 * cos_theta**5 - 20 * cos_theta**3 + 5 * cos_theta
        else:
            # Recursive Chebyshev (rarely used in practice)
            T0 = torch.ones_like(cos_theta)
            T1 = cos_theta
            for _ in range(2, self.m + 1):
                T2 = 2 * cos_theta * T1 - T0
                T0, T1 = T1, T2
            return T1
