import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weighted_saliency(input_, target, eps=1e-15):
    saliency_loss = -1.12 * target * torch.log(input_ + eps) - 1 * (1 - target) * torch.log(1 - input_ + eps)
    return torch.mean(saliency_loss)

class EdgeHoldLoss(nn.Module):
    def __init__(self, device):
        super(EdgeHoldLoss, self).__init__()
        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float,
                                             requires_grad=False).view(1, 1, 3, 3).to(device)
        # !!!!begining=1

    def forward(self, y_pred, y_gt, alpha_sal=0.7):
        # edge maps from groundTruth and predicted one
        y_gt_edges = F.relu(torch.tanh(F.conv2d(y_gt, self.laplacian_kernel, padding=(1, 1))))
        y_pred_edges = F.relu(torch.tanh(F.conv2d(y_pred, self.laplacian_kernel, padding=(1, 1))))

        sal_loss = weighted_saliency(input_=y_pred, target=y_gt)
        edge_loss = torch.mean(F.binary_cross_entropy(input=torch.sigmoid(y_pred_edges), target=y_gt_edges))

        total_loss = alpha_sal * sal_loss + (1 - alpha_sal) * edge_loss
        return total_loss

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    dummy_input = torch.autograd.Variable(torch.sigmoid(torch.randn(2, 1, 8, 16)), requires_grad=True).to(device)
    dummy_gt = torch.autograd.Variable(torch.ones_like(dummy_input)).to(device)
    print('Input Size :', dummy_input.size())

    criteria = EdgeHoldLoss(device=device)
    loss = criteria(dummy_input, dummy_gt)
    print('Loss Value :', loss)
