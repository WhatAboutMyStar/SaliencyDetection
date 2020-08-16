import torch

def accuracy(y_pred, y_true):
    return (y_pred.round() == y_true).float().mean()


def precision(y_pred, y_true):
    return torch.mul(y_pred.round(), y_true).sum() / y_pred.round().sum()


def recall(y_pred, y_true):
    return torch.mul(y_pred.round(), y_true).sum() / y_true.sum()
