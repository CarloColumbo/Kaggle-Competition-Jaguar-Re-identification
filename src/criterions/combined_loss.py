from torch import nn


class CombinedLoss(nn.Module):
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
