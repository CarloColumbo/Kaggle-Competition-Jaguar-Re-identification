from torch import nn


class CombinedLoss(nn.Module):
    """
    Combined Loss implementation.

    This loss combines two different loss functions with a weighting factor.

    The total loss is computed as:
        L = L1 + lambda_center * L2

    where:
        - L1 is the first loss function
        - L2 is the second loss function
        - lambda_center is a weighting factor for the second loss

    Args:
        first_loss (nn.Module): The first loss function.
        second_loss (nn.Module): The second loss function.
        lambda_center (float): Weighting factor for the second loss. Default is 0.1.
    """
    def __init__(self, first_loss, second_loss, lambda_center=0.1):
        super().__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.lambda_center = lambda_center

    def forward(self, features, labels):
        loss_first = self.first_loss(features, labels)
        loss_second = self.second_loss(features, labels)
        total_loss = loss_first + self.lambda_center * loss_second
        return total_loss
