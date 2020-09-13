import torch

def accuracy(y_pred, y_true):
    # y_pred = torch.sigmoid(y_pred)
    return (y_pred.round() == y_true).float().mean()


def precision(y_pred, y_true):
    # y_pred = torch.sigmoid(y_pred)
    return torch.mul(y_pred.round(), y_true).sum() / y_pred.round().sum()


def recall(y_pred, y_true):
    # y_pred = torch.sigmoid(y_pred)
    return torch.mul(y_pred.round(), y_true).sum() / y_true.sum()

def F_Measure(pre, rec):
    return (1 + 0.3) * pre * rec / (0.3 * pre + rec)

if __name__ == '__main__':
    a = torch.tensor([[0.1, 0.1, 0.2, 0.8],
                      [0.1, 0.2, 0.3, 0.4],
                      [0.5, 0.6, 0.7, 0.8],
                      [0.9, 1., 0.3, 0.4]])
    b = torch.ones(4, 4)

    # print(torch.tensor(a>=0.3, dtype=torch.float32))
    a = (a > 0.3).float()
    print(a)
    print(precision(a, b))
    print(recall(a, b))